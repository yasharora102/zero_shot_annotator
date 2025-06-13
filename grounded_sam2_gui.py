# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import threading

# from grounded_sam2_tool import annotate_image  # Make sure annotate_image(prompt=...) is updated
# from convert_to_coco import convert_to_coco
# from convert_to_yolo import convert_to_yolo # MODIFIED: Import the new conversion function


# class GroundedSAM2GUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("GroundedSAM2 GUI Tool")
#         self.image_paths = []
#         self.prompt = tk.StringVar(value="food.")

#         tk.Label(root, text="Text Prompt:").pack()
#         self.prompt_entry = tk.Entry(root, textvariable=self.prompt, width=50)
#         self.prompt_entry.pack(pady=5)

#         tk.Button(root, text="Select Images", command=self.select_images).pack(pady=5)
#         tk.Button(root, text="Run Annotation", command=self.run_annotation).pack(pady=5)
#         # tk.Button(root, text="Convert to CVAT COCO", command=self.convert_to_coco).pack(pady=5)
#         tk.Button(root, text="Convert to YOLO for CVAT", command=self.convert_to_yolo_format).pack(pady=5)

#         self.status = tk.Label(root, text="Idle", fg="blue")
#         self.status.pack(pady=10)

#     def select_images(self):
#         files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png")])
#         if files:
#             self.image_paths = list(files)
#             self.status.config(text=f"{len(files)} images selected.")

#     def run_annotation(self):
#         if not self.image_paths:
#             messagebox.showwarning("No images", "Please select image files first.")
#             return
#         self.status.config(text="Running GroundedSAM2...")

#         def process():
#             for img_path in self.image_paths:
#                 annotate_image(img_path, self.prompt.get())
#             self.status.config(text="✅ Annotation complete!")

#         threading.Thread(target=process).start()

#     def convert_to_coco(self):
#         self.status.config(text="Converting to COCO format...")

#         def process():
#             convert_to_coco()
#             self.status.config(text="✅ CVAT COCO file saved.")

#         threading.Thread(target=process).start()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = GroundedSAM2GUI(root)
#     root.mainloop()


import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

from grounded_sam2_tool import annotate_image
from convert_to_yolo import convert_to_yolo # MODIFIED: Import the new conversion function
from convert_to_coco import convert_to_coco
class GroundedSAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GroundedSAM2 GUI Tool")
        self.image_paths = []
        self.prompt = tk.StringVar(value="food.")

        tk.Label(root, text="Text Prompt:").pack()
        self.prompt_entry = tk.Entry(root, textvariable=self.prompt, width=50)
        self.prompt_entry.pack(pady=5)

        tk.Button(root, text="Select Images", command=self.select_images).pack(pady=5)
        tk.Button(root, text="Run Annotation", command=self.run_annotation).pack(pady=5)
        tk.Button(root, text="Convert to COCO for CVAT", command=self.convert_to_coco_format).pack(pady=5)
        # MODIFIED: Changed the button text and command
        tk.Button(root, text="Convert to YOLO for CVAT", command=self.convert_to_yolo_format).pack(pady=5)

        self.status = tk.Label(root, text="Idle", fg="blue")
        self.status.pack(pady=10)

    def select_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)
            self.status.config(text=f"{len(files)} images selected.")

    def run_annotation(self):
        if not self.image_paths:
            messagebox.showwarning("No images", "Please select image files first.")
            return
        self.status.config(text="Running GroundedSAM2...")

        def process():
            for img_path in self.image_paths:
                annotate_image(img_path, self.prompt.get())
            self.status.config(text="✅ Annotation complete!")

        threading.Thread(target=process).start()

    # MODIFIED: Renamed function and updated status messages
    def convert_to_coco_format(self):
        self.status.config(text="Converting to COCO format...")

        def process():
            convert_to_coco()  # Call the new conversion function
            self.status.config(text="✅ CVAT COCO file saved.")
        
        threading.Thread(target=process).start()
        
    def convert_to_yolo_format(self):
        self.status.config(text="Converting to YOLO format...")

        def process():
            convert_to_yolo()
            self.status.config(text="✅ YOLO dataset for CVAT saved!")
        
        threading.Thread(target=process).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = GroundedSAM2GUI(root)
    root.mainloop()