;; Read option<expert-normalization> across the real 1.10 component ABI.
;; Returns 0 for none and 10 + the enum discriminant for some.
(component
  (import "magnetar:model-component-graph/model-config@1.10.0" (instance $cfg
    (type $normalization (enum "never" "multiple-experts" "always"))
    (export "expert-normalization-mode" (type $normalization_e (eq $normalization)))
    (export "expert-normalization" (func (result (option $normalization_e))))))
  (core module $memory (memory (export "memory") 1))
  (core instance $mem (instantiate $memory))
  (alias core export $mem "memory" (core memory $linear))
  (core func $normalization (canon lower (func $cfg "expert-normalization") (memory $linear)))
  (core module $probe
    (import "" "memory" (memory 1))
    (import "" "normalization" (func $normalization (param i32)))
    (func (export "run") (result i32)
      i32.const 0
      call $normalization
      i32.const 0
      i32.load8_u
      i32.const 10
      i32.mul
      i32.const 1
      i32.load8_u
      i32.add))
  (core instance $i (instantiate $probe (with "" (instance
    (export "memory" (memory $linear))
    (export "normalization" (func $normalization))))))
  (func (export "run") (result u32) (canon lift (core func $i "run")))
)
