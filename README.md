> **Fork note (codeNick7):** this fork adds two build fixes so the library compiles and links on the
> **Arduino UNO Q** (STM32U585 MCU, Arduino Zephyr core `arduino:zephyr` 1.1.0), without changing behaviour
> on other boards:
>
> 1. `system_setup.cpp`: `ulong` → `unsigned long` (the Zephyr core has no `ulong` typedef).
> 2. `library.properties`: `dot_a_linkage=false`. Three source files share a basename
>    (`kernel_util.cpp`, `portable_tensor_utils.cpp`, `tensor_utils.cpp`); packed into one `.a` archive,
>    one object of each pair is dropped and linking fails with undefined `tflite::` references.
>
> Verified on an UNO Q with EloquentTinyML 3.0.1: a 1,155-parameter transformer gives the same outputs as
> TensorFlow Lite on a laptop, at ~1.96 ms per inference. Project:
> [edge-transformer-pytorch-tensorflow](https://github.com/codeNick7/edge-transformer-pytorch-tensorflow).
> Everything else is the upstream library by Simone Salerno (MIT).

Only working with Arduino Nano 33 BLE Sense.