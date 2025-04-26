from gui.GUI import *
from InterpFlow import InterpFlowModel
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

    def import_func(self, video_dr, output_folder, progress_callback):
        print("Video path:", video_dr)
        print("Output folder:", output_folder)

        batch = 12
        width = 680
        height = 480
        path_to_trt = None  # Optional TensorRT

        # Call your model here
        self.m.predict(
            video_dr=video_dr,
            output_folder=output_folder,
            batch=batch,
            output_width=width,
            output_height=height,
            path_to_trt=path_to_trt,
            progress_callback=progress_callback
        )
App()
