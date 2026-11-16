;; Imports `model-config@1.7.0`'s `multi-head-latent-attention`,
;; `mixture-of-experts-layers`, `expert-routing` and `rope-scaling` (with its
;; `yarn` case) and exports one function per import, each reading the value
;; back through the canonical ABI and returning a checksum:
;;   mla:     some * 1e9 + q-lora-rank + kv-lora-rank * 1e4 + qk-rope * 1e5
;;            + qk-nope + v-head-dim
;;   layers:  some * 10_000 + length * 100 + number of `true` entries
;;   routing: some * 1e9 + scoring * 1e8 + floor(factor) * 1e7 + count * 1e5
;;            + per-token * 1e3 + group score * 100 + selection-bias * 10
;;            + shared-expert-gated
;;   rope:    some * 1e9 + case * 1e8 + floor(factor) * 1e6 + truncate * 1e5
;;            + mscale-all-dim present * 1e4 + original positions
;; proving the `1.7.0` records, enums and the new variant case cross the real
;; host-import boundary.
(component
    (import "magnetar:model-component-graph/model-config@1.7.0" (instance $cfg
        (type $mla (record
            (field "q-lora-rank" (option u32))
            (field "kv-lora-rank" u32)
            (field "qk-nope-head-dim" u64)
            (field "qk-rope-head-dim" u64)
            (field "v-head-dim" u64)))
        (export "mla-config" (type $mla_e (eq $mla)))
        (type $scoring (enum "softmax" "sigmoid"))
        (export "expert-scoring" (type $scoring_e (eq $scoring)))
        (type $gscore (enum "max" "top-two-sum"))
        (export "expert-group-score" (type $gscore_e (eq $gscore)))
        (type $groups (record
            (field "count" u32)
            (field "per-token" u32)
            (field "score" $gscore_e)))
        (export "expert-groups" (type $groups_e (eq $groups)))
        (type $routing (record
            (field "scoring" $scoring_e)
            (field "groups" (option $groups_e))
            (field "routed-scaling-factor" float64)
            (field "selection-bias" bool)
            (field "shared-expert-gated" bool)))
        (export "expert-routing-config" (type $routing_e (eq $routing)))
        (type $banded (record
            (field "factor" float64)
            (field "low-freq-factor" float64)
            (field "high-freq-factor" float64)
            (field "original-max-position-embeddings" u64)))
        (export "banded-rope-scaling" (type $banded_e (eq $banded)))
        (type $yarn (record
            (field "factor" float64)
            (field "original-max-position-embeddings" u64)
            (field "beta-fast" float64)
            (field "beta-slow" float64)
            (field "truncate" bool)
            (field "attention-factor" float64)
            (field "mscale-all-dim" (option float64))))
        (export "yarn-rope-scaling" (type $yarn_e (eq $yarn)))
        (type $scaling (variant (case "banded" $banded_e) (case "yarn" $yarn_e)))
        (export "rope-frequency-scaling" (type $scaling_e (eq $scaling)))
        (export "multi-head-latent-attention" (func (result (option $mla_e))))
        (export "mixture-of-experts-layers" (func (result (option (list bool)))))
        (export "expert-routing" (func (result (option $routing_e))))
        (export "rope-scaling" (func (result (option $scaling_e))))))
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
    (core func $mla (canon lower (func $cfg "multi-head-latent-attention")
        (memory $linear) (realloc $realloc)))
    (core func $layers (canon lower (func $cfg "mixture-of-experts-layers")
        (memory $linear) (realloc $realloc)))
    (core func $routing (canon lower (func $cfg "expert-routing")
        (memory $linear) (realloc $realloc)))
    (core func $rope (canon lower (func $cfg "rope-scaling")
        (memory $linear) (realloc $realloc)))
    (core module $probe
        (import "" "memory" (memory 1))
        (import "" "mla" (func $mla (param i32)))
        (import "" "layers" (func $layers (param i32)))
        (import "" "routing" (func $routing (param i32)))
        (import "" "rope" (func $rope (param i32)))
        ;; option<mla-config> at 0: discriminant 0, record at 8: q-lora-rank
        ;; (discriminant 8, value 12), kv-lora-rank 16, qk-nope 24, qk-rope 32,
        ;; v 40 (the test's small values keep the sum below 2^32).
        (func (export "mla") (result i32)
            i32.const 0
            call $mla
            i32.const 0
            i32.load8_u
            i32.const 1000000000
            i32.mul
            i32.const 12
            i32.load
            i32.add
            i32.const 16
            i32.load
            i32.const 10000
            i32.mul
            i32.add
            i32.const 32
            i64.load
            i32.wrap_i64
            i32.const 100000
            i32.mul
            i32.add
            ;; + qk-nope + v-head-dim (both below 1e4 in the test)
            i32.const 24
            i64.load
            i32.wrap_i64
            i32.add
            i32.const 40
            i64.load
            i32.wrap_i64
            i32.add)
        ;; option<list<bool>> at 0: discriminant, pointer 4, length 8.
        (func (export "layers") (result i32)
            (local $ptr i32) (local $len i32) (local $i i32) (local $true i32)
            i32.const 0
            call $layers
            i32.const 4
            i32.load
            local.set $ptr
            i32.const 8
            i32.load
            local.set $len
            block $done
                loop $each
                    local.get $i
                    local.get $len
                    i32.ge_u
                    br_if $done
                    local.get $true
                    local.get $ptr
                    local.get $i
                    i32.add
                    i32.load8_u
                    i32.add
                    local.set $true
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $each
                end
            end
            i32.const 0
            i32.load8_u
            i32.const 10000
            i32.mul
            local.get $len
            i32.const 100
            i32.mul
            i32.add
            local.get $true
            i32.add)
        ;; option<expert-routing-config> at 0: discriminant 0, record at 8:
        ;; scoring 8, groups (discriminant 12, count 16, per-token 20,
        ;; score 24), routed-scaling-factor 32, selection-bias 40,
        ;; shared-expert-gated 41.
        (func (export "routing") (result i32)
            i32.const 0
            call $routing
            i32.const 0
            i32.load8_u
            i32.const 1000000000
            i32.mul
            i32.const 8
            i32.load8_u
            i32.const 100000000
            i32.mul
            i32.add
            i32.const 32
            f64.load
            i32.trunc_f64_u
            i32.const 10000000
            i32.mul
            i32.add
            i32.const 16
            i32.load
            i32.const 100000
            i32.mul
            i32.add
            i32.const 20
            i32.load
            i32.const 1000
            i32.mul
            i32.add
            i32.const 24
            i32.load8_u
            i32.const 100
            i32.mul
            i32.add
            i32.const 40
            i32.load8_u
            i32.const 10
            i32.mul
            i32.add
            i32.const 41
            i32.load8_u
            i32.add)
        ;; option<rope-frequency-scaling> at 0: discriminant 0, variant at 8:
        ;; case 8, yarn record at 16: factor 16, original 24, beta-fast 32,
        ;; beta-slow 40, truncate 48, attention-factor 56, mscale-all-dim
        ;; (discriminant 64, value 72).
        (func (export "rope") (result i32)
            i32.const 0
            call $rope
            i32.const 0
            i32.load8_u
            i32.const 1000000000
            i32.mul
            i32.const 8
            i32.load8_u
            i32.const 100000000
            i32.mul
            i32.add
            i32.const 16
            f64.load
            i32.trunc_f64_u
            i32.const 1000000
            i32.mul
            i32.add
            i32.const 48
            i32.load8_u
            i32.const 100000
            i32.mul
            i32.add
            i32.const 64
            i32.load8_u
            i32.const 10000
            i32.mul
            i32.add
            i32.const 24
            i64.load
            i32.wrap_i64
            i32.add))
    (core instance $i (instantiate $probe
        (with "" (instance
            (export "memory" (memory $linear))
            (export "mla" (func $mla))
            (export "layers" (func $layers))
            (export "routing" (func $routing))
            (export "rope" (func $rope))))))
    (func (export "mla") (result u32) (canon lift (core func $i "mla")))
    (func (export "layers") (result u32) (canon lift (core func $i "layers")))
    (func (export "routing") (result u32) (canon lift (core func $i "routing")))
    (func (export "rope") (result u32) (canon lift (core func $i "rope")))
)
