(component
    (import "example:test/host@1.0.0" (instance $host
        (export "fail" (func))))
    (core module $m
        (import "" "fail" (func $fail))
        (func (export "run")
            (call $fail)))
    (core func $fail (canon lower (func $host "fail")))
    (core instance $i
        (instantiate $m
            (with "" (instance
                (export "fail" (func $fail))))))
    (func (export "run")
        (canon lift (core func $i "run")))
)
