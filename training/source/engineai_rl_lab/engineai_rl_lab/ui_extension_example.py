# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.ext


def some_public_function(x: int):
    print("[engineai_rl_lab] some_public_function was called with x: ", x)
    return x**x


class ExampleExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[engineai_rl_lab] startup")

        self._count = 0

        self._window = omni.ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                label = omni.ui.Label("")

                def on_click():
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    self._count = 0
                    label.text = "empty"

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        print("[engineai_rl_lab] shutdown")