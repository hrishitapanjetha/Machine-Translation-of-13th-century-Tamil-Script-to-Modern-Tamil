import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import image_functions

def main():
    # Create the main window
    root = tk.Tk()
    root.title("My Application")
    
    # Set the window size to maximum
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    # Function to open file dialog and display selected image
    global file_name

    def open_image():
        current_dir = os.getcwd()
        file_path = filedialog.askopenfilename(initialdir=current_dir, filetypes=[("JPEG files", "*.jpg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((500, 500))
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            global file_name
            file_name = file_path.split('/')[-1]

            cv2_image = cv2.imread(file_path)
            cv2_preprocessed_image = image_functions.preprocess(cv2_image)
            
            segmented_image = image_functions.segment(cv2_preprocessed_image, file_name)
            segmented_image = Image.fromarray(segmented_image)
            segmented_image = segmented_image.resize((500, 500))
            segmented_photo = ImageTk.PhotoImage(segmented_image)
            segmented_image_label.configure(image=segmented_photo)
            segmented_image_label.image = segmented_photo

            # in image holder label, display the first segmented image
            holder_image = Image.open(f"./Dataset/{file_name}/segmented/0.png")
            # holder_image = holder_image.resize((500, 500))
            holder_photo = ImageTk.PhotoImage(holder_image)
            image_holder_label.configure(image=holder_photo)
            image_holder_label.image = holder_photo
            image_holder_label.label = "0.png"

    def move_next():
        # put the next segmented image in the image holder label
        # get the current image name
        current_image_name = f"./Dataset/{file_name}/segmented/{str(int(image_holder_label.label.split('/')[-1].split('.')[0]) + 1)}.png"
        holder_image = Image.open(current_image_name)
        holder_photo = ImageTk.PhotoImage(holder_image)
        image_holder_label.configure(image=holder_photo)
        image_holder_label.image = holder_photo
        image_holder_label.label = current_image_name
        

    # Function for reject button
    def reject_image():
        move_next()
    
    # Function for accept button
    def accept_image():
        move_next()
    
    # Create a button to open file dialog
    button = tk.Button(root, text="Open Image", command=open_image)
    button.grid(row=0, column=0, sticky=tk.W)  # Left-align the button
    
    # Create a label to display the selected image
    image_label = tk.Label(root)
    image_label.grid(row=2, column=0, sticky=tk.W, pady=10)  # Left-align horizontally and center-align vertically
    
    # Create a label to display the segmented image
    segmented_image_label = tk.Label(root)
    segmented_image_label.grid(row=2, column=1, sticky=tk.W, pady=10)  # Left-align horizontally and center-align vertically
    
    # Create a reject button
    reject_button = tk.Button(root, text="Reject", fg="red", command=reject_image, width=10, height=2)
    reject_button.grid(row=3, column=0, sticky=tk.W, pady=10)  # Left-align the button
    
    # Create an image holder label
    image_holder_label = tk.Label(root)
    image_holder_label.grid(row=3, sticky = tk.W, column=1, pady=10)  
    
    # Create an accept button
    accept_button = tk.Button(root, text="Accept", fg="green", command=accept_image, width=10, height=2)
    accept_button.grid(row=3, column=2, sticky=tk.E, pady=10)  # Right-align the button
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()