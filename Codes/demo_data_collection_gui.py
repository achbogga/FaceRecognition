import os
import tkinter as tk # python 3
from tkinter import *
from tkinter import messagebox
from tkinter import font  as tkfont # python 3
import webbrowser
import re
from numpy import loadtxt
import numpy as np
from tkinter import Widget

import numpy as np
import cv2
import time

#import Tkinter as tk     # python 2
#import tkFont as tkfont  # python 2

class webcam:
    def __init__(self):
        pass
    def draw_text(self, frame, text, x, y, color=(255,0,255), thickness=4, size=3):
        if x is not None and y is not None:
            cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    def cap_video(self, length = 10, output_file = 'output.avi', fps = 30.0, resolution = (640,480), prepare = True, prep_time = 8):
        #timeout = time.time() + 11   # 10 seconds from now
        cap = cv2.VideoCapture(0)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_file,fourcc, fps, resolution)
        init_time = time.time()
        if (prepare):
            final_timeout = init_time+length+prep_time+1
            test_timeout = init_time+prep_time+1
            counter_timeout_text = init_time+1
            counter_timeout = init_time+1
            counter = prep_time
        else:
            final_timeout = init_time+length+1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                #frame = cv2.flip(frame,0)
                # write the flipped frame
                if (prepare):
                    center_x = int(frame.shape[0]/2)
                    center_y = int(frame.shape[0]/2)
                    if (time.time() > counter_timeout_text and time.time() < test_timeout):
                        self.draw_text(frame, str(counter), center_x, center_y)
                        counter_timeout_text+=(1/fps)
                    if (time.time() > counter_timeout and time.time() < test_timeout):
                        counter-=1
                        counter_timeout+=1
                if ((prepare and (time.time() > test_timeout)) or (not prepare)):
                    out.write(frame)
                cv2.imshow('frame', frame)
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                    break
            else:
                break
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

class SampleApp(tk.Tk):

    def __init__(self, text_datafile, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        #GUI params to control styling
        self.bg_color = 'white'
        self.font_for_titles = ('Arial',14, 'bold')
        self.font_for_text = ('Arial',12)
        
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        
        self.scr_w = self.winfo_screenwidth()
        self.scr_h = self.winfo_screenheight()
        self.text_wrap_l = self.scr_w - 20
        self.fg_color = 'black'
        self.padx = 10
        self.configure(bg=self.bg_color)
        
        container.pack(side="top", fill="both", expand=True)
        container.configure(bg=self.bg_color)
        #container.wm_title("Welcome to OVL FR Data Collection Program!")
        self.geometry("%dx%d+0+0" % (self.scr_w, self.scr_h))
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            #frame.geometry("%dx%d+0+0" % (self.scr_w, self.scr_h))
            frame.configure(bg=self.bg_color)
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.text_datafile = text_datafile
        self.bsd = np.vectorize(self.byte_string_decoder, otypes = ['str'])
        self.update_textdb(self.text_datafile)

        self.show_frame("StartPage")

    def makeform(self, frame, fields):
        entries = []
        for field in fields:
            row = Frame(frame, bg=self.bg_color)
            lab = Label(row, bg=self.bg_color, width=35, text=field, anchor='w')
            ent = Entry(row, width=15, bg=self.bg_color)
            if field == 'Passwd (2 eng words with a single space)':
                ent.configure(show="*")
            row.pack(side=TOP, fill=X, padx=self.padx, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT, padx=self.padx, expand=YES, fill=X)
            entries.append((field, ent))
        return entries
    
    def callback(self, event):
        webbrowser.open_new(r"http://www.alarm.com/privacy_policy.aspx")
    
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    
    def make_text(self, frame, text, font = ('Arial',14,'bold')):
        lab = Label(frame, fg='black', bg=self.bg_color, wraplength = self.text_wrap_l, justify=LEFT, font=font, text=text)
        lab.pack(side=TOP, padx=self.padx, fill=X, pady=10)
    
    def display_empty_mesg (self, field):
        messagebox.showerror(field, field+' field cannot be empty!')
    
    def display_invalid_msg (self, field, text):
        messagebox.showerror(field, text+'is not a valid '+field+', Please retry with a valid '+field+'!')
    
    def byte_string_decoder(self, S):
        return S.decode('ASCII')
    
    def update_textdb(self, text_datafile):
        self.temp_ar = loadtxt(text_datafile, dtype='S50', delimiter=',')
        if not self.temp_ar.size:
            self.temp_ar = np.empty([1, 5], dtype='S50')
        self.temp_ar = self.bsd(self.temp_ar)
        temp_keys = self.temp_ar[:,0]
        temp_values = self.temp_ar[:,1:]
        self.text_db = dict(zip(temp_keys, temp_values))
        return self.text_db, self.temp_ar
    
    def upload_video(self, username):
        output_db = os.path.join(os.getcwd(),'db')
        output_dir = os.path.join(output_db,username)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        num = str(len(os.listdir(output_dir))+1)
        output_video_file = os.path.join(output_dir, '_'+num+'.avi')
        camera = webcam()
        camera.cap_video(length = 10, output_file = output_video_file, fps = 30.0, resolution = (640,480), prepare = True, prep_time = 5)
        messagebox.showinfo('video upload', 'Successful...!')    
            
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.controller.make_text(self, text="Alarm.com Employee Facial Recognition Research and Development Data Collection Program")

        button1 = tk.Button(self, bg=controller.bg_color, text="Sign In",
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, bg=controller.bg_color, text="Sign Up",
                            command=lambda: controller.show_frame("PageTwo"))
        button1.pack()
        button2.pack()
        
        self.grand_parent_name = parent.winfo_parent()
        b2 = Button(self,bg = controller.bg_color, text='Quit', fg='red', command=parent._nametowidget(self.grand_parent_name).destroy)

        b2.pack(padx=5, pady=5)


class PageOne(tk.Frame):
    '''Creates sign in page'''
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Sign In", bg=controller.bg_color, font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        fields = ['Username','Passwd (2 eng words with a single space)']
        ents = controller.makeform(self, fields)
        b1 = Button(self, bg = controller.bg_color, text='Upload another video',
              command=(lambda e=ents: self.validate_and_submit(e)))
        b1.pack(side = TOP, padx=5, pady=5)
        button = tk.Button(self, bg=controller.bg_color, text="Go to the home page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        self.grand_parent_name = parent.winfo_parent()
        b2 = Button(self,bg = controller.bg_color, text='Quit', fg='red', command=parent._nametowidget(self.grand_parent_name).destroy)
        b2.pack(padx=5, pady=5)
    def validate_and_submit(self, entries):
        flag = 1
        text_entry_list = []
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            text_entry_list.append(text)
            if (text=="" or text==None):
                print('%s: should not be empty!' % (field))
                flag = 0
                self.controller.display_empty_mesg(field)
            elif(field == 'Username'):
                if (text not in self.controller.text_db):
                    flag = 0
                    messagebox.showerror(field+' error!' , 'Username: '+ text +' is not registered! Please sign up..!')
                    self.controller.show_frame("PageTwo")
            elif(field == 'Passwd (2 eng words with a single space)'):
                if (flag and not (text == self.controller.text_db[text_entry_list[0]][0])):
                    flag = 0
                    messagebox.showerror(' Password incorrect!' , 'Please try again!')
        if flag:
            self.controller.upload_video(text_entry_list[0])


class PageTwo(tk.Frame):
    '''Creates sign up page'''
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        
        
        label = tk.Label(self, text="Sign up", bg=controller.bg_color,  font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        
        self.controller.make_text(self, "Alarm.com Employee Facial Recognition Consent for Research and Development (dated as of June 9, 2017)", font=('Arial', 14, 'bold'))
        self.controller.make_text(self, "By clicking “I Agree” or signing below, I consent to Alarm.com’s collection, transmission, maintenance, processing, and use of my video images and biometric data (collectively, “Facial Recognition Data”) in order to enable the research and development of Alarm.com’s facial recognition services, as described in this notice and in accordance with Alarm.com’s Privacy Policy. I further acknowledge and agree that Alarm.com may share my Facial Recognition Data with its affiliates for use in accordance with this notice and that Alarm.com and such affiliates may retain my Facial Recognition Data. I acknowledge and agree that I am 13 years old or older, currently an employee of Alarm.com and not a resident of the State of Illinois.  Please read our Privacy Policy and the acknowledgement below, and click “I Agree”, or sign below, to consent.",font=('Arial', 12))
        self.controller.make_text(self, "I have reviewed this notice, the Alarm.com Privacy Policy and hereby consent to the collection, use and disclosure of my Facial Recognition Data in accordance with the Privacy Policy and this notice.", font = ('Arial',12,'italic'))
        fields = ['Username','Passwd (2 eng words with a single space)', 'Last Name', 'First Name', 'Email id']
        ents = controller.makeform(self, fields)
        
        row = Frame(self, bg=controller.bg_color)
        agree = IntVar()
        c = Checkbutton(row, bg= controller.bg_color, text="I agree and give consent",variable=agree)
        row.pack(side=TOP, padx = (5,5), pady=(5,5))
        c.pack(side=LEFT)
        
        self.bind('<Return>', (lambda event, e=ents: self.validate_and_submit(e, agree)))
        
        link = Label(row, text="Alarm.com Privacy Policy", fg="blue", bg=controller.bg_color, cursor="hand2")
        link.pack(side = LEFT, padx = (100,10))
        link.bind("<Button-1>", controller.callback)
        
        b1 = Button(self, bg = controller.bg_color, text='Submit',
              command=(lambda e=ents: self.validate_and_submit(e, agree)))
        b1.pack(side = TOP, padx=5, pady=5)
        self.grand_parent_name = parent.winfo_parent()
        b2 = Button(self,bg = controller.bg_color, text='Quit', fg='red', command=parent._nametowidget(self.grand_parent_name).destroy)
        b2.pack(padx=5, pady=5)
        
        
        button = tk.Button(self, bg=controller.bg_color, text="Go to the home page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
    
    def submit_new_entry_to_textdb(self, entries):
        self.controller.temp_ar = np.vstack((self.controller.temp_ar,np.asarray(entries, dtype='S50')))
        np.savetxt(self.controller.text_datafile, self.controller.temp_ar, fmt=('%s','%s','%s','%s','%s'), delimiter=',')
        self.controller.update_textdb(self.controller.text_datafile)
    
    def validate_and_submit(self, entries, agree):
        flag = 1
        text_entry_list = []
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            text_entry_list.append(text)
            if (text=="" or text==None):
                print('%s: should not be empty!' % (field))
                flag = 0
                self.controller.display_empty_mesg(field)
            elif(field == 'Username'):
                if (text in self.controller.text_db):
                    flag = 0
                    messagebox.showerror(field+' error!' , 'Username: '+ text +' already exists! Please enter another username to retry!')
            elif (field == 'Email id' and not re.match(r"[^@]+@[^@]+\.[^@]+", text)):
                print('%s: is not a valid email! Please enter a valid email id!' % (text))
                flag=0
                self.controller.display_invalid_msg(field, text)
            #^[a-zA-Z]+\s[a-zA-Z]+$
            elif (field == 'Passwd (2 eng words with a single space)' and not re.match(r"^[a-zA-Z]+\s[a-zA-Z]+$", text)):
                print('The Passphrase entered is not a valid one! Please enter two english words separated with only one space')
                flag=0
                self.controller.display_invalid_msg(field, 'entered passphrase')
        if flag and agree.get():
            self.submit_new_entry_to_textdb(text_entry_list)
            messagebox.showinfo('Registration status', 'Successful...!\nPlease read the following instructions carefully and follow them!\n1. Now your face video will be recorded for 10 seconds after 5 seconds of preparation time.\n2. Please adjust your face relative to the position of the webcamera so that your face is in the center.\n')
            self.controller.upload_video(text_entry_list[0])
            self.controller.show_frame("StartPage")
            print("Agreed and submitted")
            #self.contorller.root.destroy()
        elif (flag and not agree.get()):
            messagebox.showerror('Agreement required', 'You have to agree to the terms to proceed!\n Please agree after reviewing the terms...')
            print("Not agreed and submitted!")
        else:
            print('flag is reset for some reason!')

if __name__ == "__main__":
    #text_db = os.path.join('text_db', 'text_data.csv')
    text_db = 'text_data.csv'
    app = SampleApp(text_db)
    app.mainloop()
