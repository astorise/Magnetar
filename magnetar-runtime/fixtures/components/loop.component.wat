(component
    (core module $m
        (func (export "run")
            (loop $again
                br $again)))
    (core instance $i (instantiate $m))
    (func (export "run")
        (canon lift (core func $i "run")))
)
