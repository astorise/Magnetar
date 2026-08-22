(component
    (import "example:test/host@1.0.0" (instance $host
        (export "ping" (func))))
    (core module $m
        (import "" "ping" (func $ping))
        (func (export "run")
            (call $ping)))
    (core func $ping (canon lower (func $host "ping")))
    (core instance $i
        (instantiate $m
            (with "" (instance
                (export "ping" (func $ping))))))
    (func (export "run")
        (canon lift (core func $i "run")))
)
