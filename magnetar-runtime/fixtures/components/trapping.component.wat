(component
    (core module $m
        (func (export "run")
            unreachable))
    (core instance $i (instantiate $m))
    (func (export "run")
        (canon lift (core func $i "run")))
)
