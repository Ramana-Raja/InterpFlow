from gui.GUI import *


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x600')
        self.title('InterpFlow')
        self.minsize(800,500)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=6)

        self.video_import = VideoImport(self, self.import_func)

        self.mainloop()

    def handle_error(self, error):
        error_message = f"Error: {str(error)}"
        print(error_message)
        self.video_import.update_status(error_message, success=False)

    def import_func(self,
            version,
            batch,
            output_width,
            output_height,
            use_to_trt,
            video_dr,
            output_folder,
            progress_callback):

        from InterpFlow import InterpFlowModel
        import os.path
        import re

        self.m = InterpFlowModel(version)

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
        try:
            self.m.predict(
                video_dr=video_dr,
                output_folder=output_folder,
                batch=int(batch),
                output_width=int(width),
                output_height=int(height),
                path_to_trt=trt,
                progress_callback=progress_callback,
                error_callback = self.handle_error
            )
        except Exception as e:
            self.handle_error(e)

if __name__ == "__main__":
    App()

