from tkinter import messagebox
from tkinter import *
import webbrowser
import re
from numpy import loadtxt
import numpy as np
class App:
    def __init__(self, master, text_datafile):
        temp_ar = loadtxt(text_datafile, dtype='S50', delimiter=',')
        temp_keys = temp_ar[:,0]
        temp_values = temp_ar[:,1:]
        self.text_db = dict(zip(temp_keys, temp_values))
        
        master.configure(bg='white')
        master.wm_title("Welcome to OVL FR Data Collection Program!")
        w, h = master.winfo_screenwidth(), master.winfo_screenheight()
        master.geometry("%dx%d+0+0" % (w, h))
        self.b = w
        self.padx_l = .13*self.b
        self.padx_r = .13*self.b
        self.padx_temp = (self.padx_l,self.padx_r)
        self.wrap_l = .40*self.b
        self.maketext(master)
        fields = 'Username', 'Last Name', 'First Name', 'Email id'
        ents = self.makeform(master, fields)
        
        r = Frame(master, bg='white')
        agree = IntVar()
        c = Checkbutton(r, bg= 'white', text="I agree and give consent",variable=agree)
        r.pack(side=TOP, padx = (5,5), pady=(5,5))
        c.pack(side=LEFT)
        
        master.bind('<Return>', (lambda event, e=ents: self.validate_and_submit(e, agree)))
        
        link = Label(r, text="Alarm.com Privacy Policy", fg="blue", bg='white', cursor="hand2")
        link.pack(side = LEFT, padx = (100,10))
        link.bind("<Button-1>", self.callback)
        
        b1 = Button(master, bg = 'white', text='Submit',
              command=(lambda e=ents: self.validate_and_submit(e, agree)))
        b1.pack(side = TOP, padx=5, pady=5)
        b2 = Button(master,bg = 'white', text='Quit', fg='red', command=master.destroy)
        b2.pack(padx=5, pady=5)
    
    def callback(self, event):
        webbrowser.open_new(r"http://www.alarm.com/privacy_policy.aspx")
    
    def display_empty_mesg (self, field):
        messagebox.showerror(field, field+' field cannot be empty!')
    
    def display_invalid_msg (self, field, text):
        messagebox.showerror(field, text+'is not a valid '+field+', Please retry with a valid '+field+'!')
    
    def validate_and_submit(self, entries, agree):
        flag = 1
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            if (text=="" or text==None):
                print('%s: should not be empty!' % (field))
                flag = 0
                self.display_empty_mesg(field)
            elif(field == 'Username'):
                if (text.encode('ASCII') in self.text_db):
                    flag = 0
                    messagebox.showerror(field+' error!' , 'Username: '+ text +' already exists! Please enter another username to retry!')
            elif (field == 'Email id' and not re.match(r"[^@]+@[^@]+\.[^@]+", text)):
                print('%s: is not a valid email! Please enter a valid email id!' % (text))
                flag=0
                self.display_invalid_msg(field, text)
        if flag and agree.get():
            print("Agreed and submitted")
        elif (flag and not agree.get()):
            messagebox.showerror('Agreement required', 'You have to agree to the terms to proceed!\n Please agree after reviewing the terms...')
            print("Not agreed and submitted!")
        else:
            print('flag is reset for some reason!')

    def makeform(self, master, fields):
        entries = []
        for field in fields:
            row = Frame(master, bg='white')
            lab = Label(row, bg='white', width=15, text=field, anchor='w')
            ent = Entry(row, width=15, bg='white')
            row.pack(side=TOP, fill=X, padx=(self.padx_l,5), pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT, padx=(0,self.padx_r), expand=YES, fill=X)
            entries.append((field, ent))
        return entries
    def maketext(self, master):
        title_text = """Alarm.com Employee Facial Recognition Consent for Research and Development (dated as of June 9, 2017)"""
        row = Frame(master, bg = 'white')
        lab = Label(row, fg='black', bg='white', wraplength = self.wrap_l, justify=LEFT, font=('Arial',14,'bold'), text=title_text, underline=True, anchor='w')
        row.pack(side=TOP, padx=self.padx_temp, fill=X, pady=5)
        lab.pack(side=TOP)
        title_text = """By clicking “I Agree” or signing below, I consent to Alarm.com’s collection, transmission, maintenance, processing, and use of my video images and biometric data (collectively, “Facial Recognition Data”) in order to enable the research and development of Alarm.com’s facial recognition services, as described in this notice and in accordance with Alarm.com’s Privacy Policy. I further acknowledge and agree that Alarm.com may share my Facial Recognition Data with its affiliates for use in accordance with this notice and that Alarm.com and such affiliates may retain my Facial Recognition Data. I acknowledge and agree that I am 13 years old or older, currently an employee of Alarm.com and not a resident of the State of Illinois.  Please read our Privacy Policy and the acknowledgement below, and click “I Agree”, or sign below, to consent."""
        row = Frame(master, bg = 'white')
        lab = Label(row, bg='white', wraplength=self.wrap_l, justify=LEFT, font=('Arial',12), text=title_text, anchor='w')
        row.pack(side=TOP, padx=self.padx_temp, fill=X, pady=5)
        lab.pack(side=TOP)
        title_text = """I have reviewed this notice, the Alarm.com Privacy Policy and hereby consent to the collection, use and disclosure of my Facial Recognition Data in accordance with the Privacy Policy and this notice."""
        row = Frame(master, bg = 'white')
        lab = Label(row, bg='white', wraplength=self.wrap_l, justify=LEFT, font=('Arial',14, 'bold'), text=title_text, anchor='w')
        row.pack(side=TOP, padx=self.padx_temp, fill=X, pady=5)
        lab.pack(side=TOP)
def register():
    root = Tk()
    app = App(root, 'text_data.csv')
    root.mainloop()
