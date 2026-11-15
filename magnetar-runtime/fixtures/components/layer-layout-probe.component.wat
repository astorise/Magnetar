;; Imports `model-config@1.6.0`'s `layer-layout` and `options` and exports
;; `run`, which reads them back through the canonical ABI and returns
;;   layers * 1_000_000 + per-layer-input-size * 1000
;;   + layer 0's sliding window + layer 1's head dimension
;;   + softcap (truncated) * 10_000_000,
;; proving the `1.6.0` records cross the real host-import boundary.
(component
    (import "magnetar:model-component-graph/model-config@1.6.0" (instance $cfg
        (type $ls (record
            (field "sliding-window" (option u32))
            (field "head-dimension" u64)
            (field "num-key-value-heads" u32)
            (field "rope-theta" float64)
            (field "partial-rotary-factor" (option float64))
            (field "value-from-key" bool)
            (field "kv-source-layer" (option u32))
            (field "intermediate-size" u64)))
        (export "layer-spec" (type $ls_e (eq $ls)))
        (type $opts (record
            (field "final-logit-softcapping" (option float64))
            (field "per-layer-input-size" (option u32))))
        (export "architecture-options" (type $opts_e (eq $opts)))
        (type $layout (option (list $ls_e)))
        (export "layer-layout" (func (result $layout)))
        (export "options" (func (result $opts_e)))))
    (core module $memory
        (memory (export "memory") 1)
        (global $next (mut i32) (i32.const 1024))
        (func (export "realloc") (param i32 i32 i32 i32) (result i32)
            (local $ptr i32)
            ;; Bump allocator: align `next` up to `align`, then advance.
            global.get $next
            local.get 2
            i32.add
            i32.const 1
            i32.sub
            i32.const 0
            local.get 2
            i32.sub
            i32.and
            local.tee $ptr
            local.get 3
            i32.add
            global.set $next
            local.get $ptr))
    (core instance $mem (instantiate $memory))
    (alias core export $mem "memory" (core memory $linear))
    (alias core export $mem "realloc" (core func $realloc))
    (core func $layer_layout (canon lower (func $cfg "layer-layout")
        (memory $linear) (realloc $realloc)))
    (core func $options (canon lower (func $cfg "options")
        (memory $linear) (realloc $realloc)))
    (core module $probe
        (import "" "memory" (memory 1))
        (import "" "layer-layout" (func $layer_layout (param i32)))
        (import "" "options" (func $options (param i32)))
        (func (export "run") (result i32)
            (local $list i32)
            ;; option<list<layer-spec>> at 0: discriminant, pointer, length.
            i32.const 0
            call $layer_layout
            ;; architecture-options at 16: option<f64> (16, payload 24),
            ;; option<u32> (32, payload 36).
            i32.const 16
            call $options
            i32.const 4
            i32.load
            local.set $list
            ;; layers * 1_000_000
            i32.const 8
            i32.load
            i32.const 1000000
            i32.mul
            ;; + per-layer-input-size * 1000
            i32.const 36
            i32.load
            i32.const 1000
            i32.mul
            i32.add
            ;; + layer 0's sliding window (payload at offset 4)
            local.get $list
            i32.load offset=4
            i32.add
            ;; + layer 1's head dimension (record size 72, field at 8)
            local.get $list
            i64.load offset=80
            i32.wrap_i64
            i32.add
            ;; + softcap * 10_000_000
            i32.const 24
            f64.load
            i32.trunc_f64_u
            i32.const 10000000
            i32.mul
            i32.add))
    (core instance $i (instantiate $probe
        (with "" (instance
            (export "memory" (memory $linear))
            (export "layer-layout" (func $layer_layout))
            (export "options" (func $options))))))
    (func (export "run") (result u32)
        (canon lift (core func $i "run")))
)
