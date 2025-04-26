import customtkinter as ctk
from tkinter import filedialog

class VideoImport(ctk.CTkFrame):
    def __init__(self, parent, import_func):
        super().__init__(master=parent)
        self.grid(column=0, columnspan=2, row=0, sticky='nsew')
        self.import_func = import_func

        # Center frame
        center_frame = ctk.CTkFrame(self)
        center_frame.pack(expand=True)

        # Buttons
        ctk.CTkButton(center_frame, text='Select Video', command=self.select_video).pack(pady=10)
        ctk.CTkButton(center_frame, text='Select Output Folder', command=self.select_output_folder).pack(pady=10)
        ctk.CTkButton(center_frame, text='Start Processing', command=self.start_processing).pack(pady=20)

        # Status label
        self.status_label = ctk.CTkLabel(center_frame, text="", text_color="red")
        self.status_label.pack(pady=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(center_frame, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        # Paths
        self.video_path = None
        self.output_folder = None

    def select_video(self):
        video_file = filedialog.askopenfile(title="Select Video")
        if video_file:
            self.video_path = video_file.name
            self.update_status(f"Selected video: {self.video_path}", success=True)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.update_status(f"Selected output folder: {self.output_folder}", success=True)

    def start_processing(self):
        if not self.video_path or not self.output_folder:
            self.update_status("Please select both input video and output folder.", success=False)
            return

        self.update_status("Processing started...", success=True)
        self.progress_bar.set(0)

        # Call the import function
        self.import_func(
            video_dr=self.video_path,
            output_folder=self.output_folder,
            progress_callback=self.update_progress
        )

    def update_status(self, message, success=True):
        """ Update the status label with a message """
        color = "green" if success else "red"
        self.status_label.configure(text=message, text_color=color)

    def update_progress(self, current, total):
        """ Update the progress bar based on current progress """
        percent = current / total
        self.progress_bar.set(percent)
        self.progress_bar.update()
