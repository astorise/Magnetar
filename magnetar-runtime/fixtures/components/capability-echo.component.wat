(component
    (import "example:test/capability@1.0.0" (instance $cap
        (export "echo" (func (param "x" u32) (result u32)))))
    (core module $m
        (import "" "echo" (func $echo (param i32) (result i32)))
        (func (export "run") (result i32)
            i32.const 41
            call $echo))
    (core func $echo (canon lower (func $cap "echo")))
    (core instance $i
        (instantiate $m
            (with "" (instance
                (export "echo" (func $echo))))))
    (func (export "run") (result u32)
        (canon lift (core func $i "run")))
)
