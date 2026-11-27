;; Verify the 1.15.0 hybrid configuration crosses the actual Component ABI,
;; including ordered enum values, u64 dimensions, float limits and flags.
;; `run` returns one only for the values in the Wasmtime test.
(component
  (import "magnetar:model-component-graph/model-config@1.15.0" (instance $cfg
    (type $block (enum "mamba2" "attention" "dense-mlp" "routed-moe"))
    (export "hybrid-decoder-block" (type $block-e (eq $block)))
    (type $hybrid (record
      (field "layers" (list $block-e))
      (field "heads" u32)
      (field "head-dimension" u64)
      (field "groups" u32)
      (field "state-dimension" u64)
      (field "conv-kernel" u32)
      (field "epsilon" float64)
      (field "dt-min" float64)
      (field "dt-max" (option float64))
      (field "conv-bias" bool)
      (field "linear-bias" bool)))
    (export "mamba2-hybrid-config" (type $hybrid-e (eq $hybrid)))
    (export "mamba2-hybrid" (func (result (option $hybrid-e))))))
  (core module $memory
    (memory (export "memory") 1)
    (global $next (mut i32) (i32.const 1024))
    (func (export "realloc") (param i32 i32 i32 i32) (result i32)
      (local $ptr i32)
      global.get $next local.get 2 i32.add i32.const 1 i32.sub
      i32.const 0 local.get 2 i32.sub i32.and local.tee $ptr
      local.get 3 i32.add global.set $next local.get $ptr))
  (core instance $mem (instantiate $memory))
  (alias core export $mem "memory" (core memory $linear))
  (alias core export $mem "realloc" (core func $realloc))
  (core func $hybrid (canon lower (func $cfg "mamba2-hybrid")
    (memory $linear) (realloc $realloc)))
  (core module $probe
    (import "" "memory" (memory 1))
    (import "" "hybrid" (func $hybrid (param i32)))
    ;; option tag at 0; record at 8; list pointer/length at 8/12;
    ;; heads 16, head dimension 24, groups 32, state dimension 40,
    ;; kernel 48, epsilon 56, dt-min 64, dt-max tag/value 72/80,
    ;; flags 88/89.
    (func (export "run") (result i32)
      (local $ptr i32)
      i32.const 0 call $hybrid
      i32.const 0 i32.load8_u i32.eqz
      if i32.const 0 return end
      i32.const 8 i32.load local.set $ptr
      i32.const 12 i32.load i32.const 4 i32.eq
      if (result i32)
        local.get $ptr i32.load8_u i32.const 0 i32.eq
        local.get $ptr i32.const 1 i32.add i32.load8_u i32.const 1 i32.eq i32.and
        local.get $ptr i32.const 2 i32.add i32.load8_u i32.const 2 i32.eq i32.and
        local.get $ptr i32.const 3 i32.add i32.load8_u i32.const 3 i32.eq i32.and
        i32.const 16 i32.load i32.const 4 i32.eq i32.and
        i32.const 24 i64.load i64.const 8 i64.eq i32.and
        i32.const 32 i32.load i32.const 2 i32.eq i32.and
        i32.const 40 i64.load i64.const 3 i64.eq i32.and
        i32.const 48 i32.load i32.const 4 i32.eq i32.and
        i32.const 56 f64.load f64.const 0.125 f64.eq i32.and
        i32.const 64 f64.load f64.const 0.25 f64.eq i32.and
        i32.const 72 i32.load8_u i32.const 1 i32.eq i32.and
        i32.const 80 f64.load f64.const 0.5 f64.eq i32.and
        i32.const 88 i32.load8_u i32.const 1 i32.eq i32.and
        i32.const 89 i32.load8_u i32.const 0 i32.eq i32.and
      else
        i32.const 0
      end))
  (core instance $i (instantiate $probe
    (with "" (instance (export "memory" (memory $linear)) (export "hybrid" (func $hybrid))))))
  (func (export "run") (result u32) (canon lift (core func $i "run")))
)
