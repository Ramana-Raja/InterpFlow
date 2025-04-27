from gui.GUI import *
from InterpFlow import InterpFlowModel
import os.path
import re

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.m = InterpFlowModel()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x600')
        self.title('InterpFlow')
        self.minsize(800,500)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=6)

        self.video_import = VideoImport(self, self.import_func)

        self.mainloop()

    def import_func(self,batch,
            output_width,
            output_height,
            use_to_trt,
            video_dr,
            output_folder,
            progress_callback):

        batch = batch
        width = output_width
        height = output_height
        trt = None
        if use_to_trt:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            trt_dir = os.path.join(script_dir, 'trained_models', 'trt_models')
            pattern = f"model_{height}_{width}_{batch}.trt"

            for root, dirs, files in os.walk(trt_dir):
             for file in files:
                if re.match(pattern, file):
                    trt = os.path.join(root, file)

        self.m.predict(
            video_dr=video_dr,
            output_folder=output_folder,
            batch=batch,
            output_width=width,
            output_height=height,
            path_to_trt=trt,
            progress_callback=progress_callback,
        )
App()
