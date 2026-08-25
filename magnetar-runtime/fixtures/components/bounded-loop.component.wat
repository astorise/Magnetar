;; Runs a fixed number of iterations and returns.
;;
;; Unlike loop.component.wat this terminates, and unlike unit-export.component.wat
;; it consumes a measurable amount of fuel. That combination is what makes a
;; per-invocation execution budget observable: the call has to both cost
;; something and leave the instance usable for the next call.
(component
    (core module $m
        (func (export "run")
            (local $i i32)
            (local.set $i (i32.const 10000))
            (loop $again
                (local.set $i (i32.sub (local.get $i) (i32.const 1)))
                (br_if $again (i32.gt_s (local.get $i) (i32.const 0))))))
    (core instance $i (instantiate $m))
    (func (export "run")
        (canon lift (core func $i "run")))
)
