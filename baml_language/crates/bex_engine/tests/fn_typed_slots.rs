//! Function-typed parameters and class fields accepting lambdas.

mod common;
use std::sync::Arc;

use bex_engine::{BexEngine, BexExternalValue, FunctionCallContextBuilder};
use bex_heap::CollectionLevel;
use common::compile_for_engine;
use sys_native::SysOpsExt;

async fn run(source: &str) -> BexExternalValue {
    let snapshot = compile_for_engine(source);
    let engine = Arc::new(
        BexEngine::new(snapshot, Arc::new(sys_native::SysOps::native()), Vec::new())
            .expect("engine"),
    );
    engine
        .call_function(
            "main",
            vec![],
            FunctionCallContextBuilder::new(sys_types::CallId::next()).build(),
            true,
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn clean_lambda_into_fn_param() {
    let r = run(r#"
        function take_f(f: (int) -> int) -> int { f(20) }
        function main() -> int { take_f((x) -> { x + 22 }) }
    "#)
    .await;
    assert_eq!(r, BexExternalValue::Int(42));
}

#[tokio::test]
async fn clean_lambda_into_class_field() {
    let r = run(r#"
        class F { f (int) -> int throws never }
        function main() -> int {
            let holder = F { f: (x: int) -> int { x * 2 } };
            let g = holder.f;
            g(21)
        }
    "#)
    .await;
    assert_eq!(r, BexExternalValue::Int(42));
}

#[tokio::test]
async fn returned_callbacks_share_captured_mutable_state() {
    let source = r#"
        class Callbacks {
            increment: () -> null throws never
            observe: () -> int throws never
        }

        class State {
            count: int
        }

        function make_callbacks() -> Callbacks {
            let state = State { count: 0 }
            Callbacks {
                increment: () -> null {
                    state.count += 1
                    null
                },
                observe: () -> int {
                    state.count
                }
            }
        }

        function increment_callback(callbacks: Callbacks) -> () -> null throws never {
            callbacks.increment
        }

        function observe_callback(callbacks: Callbacks) -> () -> int throws never {
            callbacks.observe
        }
    "#;
    let snapshot = compile_for_engine(source);
    let engine = Arc::new(
        BexEngine::new(snapshot, Arc::new(sys_native::SysOps::native()), Vec::new())
            .expect("engine"),
    );
    let context = || FunctionCallContextBuilder::new(sys_types::CallId::next()).build();

    let callbacks = engine
        .call_function("make_callbacks", vec![], context(), false)
        .await
        .expect("callbacks should be returned as a rooted handle");
    assert!(matches!(callbacks, BexExternalValue::Handle(_)));

    let increment = engine
        .call_function(
            "increment_callback",
            vec![callbacks.clone()],
            context(),
            false,
        )
        .await
        .expect("increment callback should be returned as a rooted handle");
    let BexExternalValue::Handle(increment) = increment else {
        panic!("increment callback should be a handle");
    };
    let observe = engine
        .call_function("observe_callback", vec![callbacks], context(), false)
        .await
        .expect("observe callback should be returned as a rooted handle");
    let BexExternalValue::Handle(observe) = observe else {
        panic!("observe callback should be a handle");
    };

    engine
        .call_callable(increment, vec![], context(), true)
        .await
        .expect("increment callback should run");
    engine.collect_garbage(CollectionLevel::Major).await;
    let observed = engine
        .call_callable(observe, vec![], context(), true)
        .await
        .expect("observe callback should run after garbage collection");

    assert_eq!(observed, BexExternalValue::Int(1));
}

// Union metadata and invoking the returned heap handle are host-boundary
// contracts: a BAML-only test never runs the outbound value converter.
#[tokio::test]
async fn returned_callable_union_preserves_function_arm_and_captures() {
    let source = r#"
        class Refused { status: int }
        type Accept = (value: int) -> int throws never;

        function named<T>(value: T) -> T throws never { value }

        class Adder {
            offset: int
            function apply(self, value: int) -> int throws never { self.offset + value }
        }

        function make_callback(kind: int, offset: int) -> Refused | Accept throws never {
            if (kind == 0) {
                Refused { status: 401 }
            } else if (kind == 1) {
                (value: int) -> int throws never { offset + value }
            } else if (kind == 2) {
                named<int>
            } else {
                Adder { offset: offset }.apply
            }
        }
    "#;
    let engine = Arc::new(
        BexEngine::new(
            compile_for_engine(source),
            Arc::new(sys_native::SysOps::native()),
            Vec::new(),
        )
        .expect("engine"),
    );
    let context = || FunctionCallContextBuilder::new(sys_types::CallId::next()).build();

    for (kind, expected) in [(1, 29), (2, 17), (3, 29)] {
        let returned = engine
            .call_function(
                "make_callback",
                vec![BexExternalValue::Int(kind), BexExternalValue::Int(12)],
                context(),
                true,
            )
            .await
            .expect("callable union should cross the host boundary");
        let BexExternalValue::Union { value, metadata } = returned else {
            panic!("expected a union envelope");
        };
        assert!(matches!(
            metadata.selected_option,
            bex_engine::RuntimeTy::Function { .. }
        ));
        let BexExternalValue::Adt(bex_external_types::BexExternalAdt::TaggedHeapHandle {
            kind: bex_external_types::TaggedHeapHandleKind::Callable,
            ty,
            heap_handle,
        }) = *value
        else {
            panic!("expected a rooted callable");
        };
        assert_eq!(ty, metadata.selected_option);
        let result = engine
            .call_callable(
                heap_handle,
                vec![BexExternalValue::Int(17)],
                context(),
                true,
            )
            .await
            .expect("returned callable should run");
        assert_eq!(result, BexExternalValue::Int(expected));
    }

    let refused = engine
        .call_function(
            "make_callback",
            vec![BexExternalValue::Int(0), BexExternalValue::Int(12)],
            context(),
            true,
        )
        .await
        .expect("non-callable union arm should remain usable");
    let BexExternalValue::Union { value, metadata } = refused else {
        panic!("expected a union envelope");
    };
    assert!(matches!(
        metadata.selected_option,
        bex_engine::RuntimeTy::Class(..)
    ));
    let BexExternalValue::Instance { fields, .. } = *value else {
        panic!("expected the refusal value");
    };
    assert_eq!(fields["status"], BexExternalValue::Int(401));
}

/// Throws semantics around fn-typed slots: an omitted `throws` on a callback
/// PARAM is effect-polymorphic (the callee forwards whatever the callback
/// throws — it does NOT mean `throws never`), and a function without a
/// `throws` clause infers its surface. Enforcement happens where a contract
/// is EXPLICIT: a `throws never` caller passing a throwing lambda through a
/// forwarding callee is rejected at compile time (E0096).
#[tokio::test]
#[should_panic(expected = "declared throws is `never`")]
async fn throwing_lambda_violates_explicit_never_contract() {
    run(r#"
        function take_f(f: (int) -> int) -> int { f(20) }
        function main() -> int throws never {
            take_f((x: int) -> int { throw "boom" })
        }
    "#)
    .await;
}
