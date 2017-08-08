#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shutil import copyfile
import tkinter as tk # python 3
from tkinter import *
from tkinter import messagebox
from tkinter import font  as tkfont # python 3
import webbrowser
import re
from numpy import loadtxt
from tkinter import Widget
import pickle

import numpy as np
import cv2
import time

class webcam:
	def __init__(self):
		pass
	def draw_text(self, frame, text, x, y, color=(255,0,255), thickness=4, size=3):
		if x is not None and y is not None:
			cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
	def cap_video(self, length = 10, output_file = 'output.avi', fps = 30.0, resolution = (640,480), prepare = True, prep_time = 8):
		#timeout = time.time() + 11   # 10 seconds from now
		cap = cv2.VideoCapture(0)
		out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)
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
		cap.release()
		out.release()
		cv2.destroyAllWindows()
	def disp_video(self, video_src, fps = 30):
		cap = cv2.VideoCapture(video_src)
		init_time = time.time()
		timeout = init_time+11
		for i in range(fps*10):
			ret, frame = cap.read()
			cv2.imshow('frame',frame)
			if (cv2.waitKey(int(1000/fps)) & 0xFF == ord('q')) or (time.time() > timeout):
				break
		cap.release()
		cv2.destroyAllWindows()



class SampleApp(tk.Tk):

	def __init__(self, text_datafile, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		#self.scr_w = self.winfo_screenwidth()
		#self.scr_h = self.winfo_screenheight()
		self.scr_w = 1366
		self.scr_h = 768
		#self.title_font = tkfont.Font(family='Helvetica', size=self.scr_w/90, weight="bold", slant="italic")
		#GUI params to control styling
		self.bg_color = 'white'
		self.font_for_titles = ('Arial',int(self.scr_w/96), 'bold')
		self.font_for_text = ('Arial',int(self.scr_w/120))
		self.font_for_buttons = ('Arial', int(self.scr_w/96), 'bold')
		self.font_for_entries = ('Arial', int(self.scr_w/90))
		
		# the container is where we'll stack a bunch of frames
		# on top of each other, then the one we want visible
		# will be raised above the others
		container = tk.Frame(self)
		
		
		self.text_wrap_l = int(self.scr_w*0.98)
		self.fg_color = 'black'
		self.text_padx = int(self.scr_w*0.0053)
		self.button_padx = int(self.scr_w*0.04)
		self.pady = int(self.scr_h*0.00652)
		#
		self.configure(bg=self.bg_color)
		
		container.pack(side="top", fill="both", expand=True)
		container.configure(bg=self.bg_color)
		#container.wm_title("Welcome to OVL FR Data Collection Program!")
		self.geometry("%dx%d+0+0" % (self.scr_w, self.scr_h))
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}
		for F in (StartPage, PageTwo):
			page_name = F.__name__
			frame = F(parent=container, controller=self)
			self.frames[page_name] = frame

			# put all of the pages in the same location;
			# the one on the top of the stacking order
			# will be the one that is visible.
			#frame.geometry("%dx%d+0+0" % (self.scr_w, self.scr_h))
			frame.configure(bg=self.bg_color)
			frame.grid(row=0, column=0, sticky="nsew")
		self.text_db = []
		self.text_datafile = text_datafile
		if not (os.stat(text_datafile).st_size == 0):
			self.update_textdb(self.text_datafile)
		self.show_frame("StartPage")

	def makeform(self, frame, fields):
		entries = []
		for field in fields:
			row = Frame(frame, bg=self.bg_color)
			lab = Label(row, bg=self.bg_color, width=int(self.scr_w*0.01823), text=field, anchor='w', font=self.font_for_entries)
			ent = Entry(row, width=int(self.scr_w*0.00782), bg=self.bg_color, font=self.font_for_entries)
			#if field == 'Password':
			#    ent.configure(show="*")
			row.pack(side=TOP, fill=X, padx=self.text_padx, pady=self.pady)
			lab.pack(side=LEFT)
			ent.pack(side=LEFT, padx=self.text_padx, expand=YES, fill=X)
			entries.append((field, ent))
		self.entries = entries
		return self.entries
	def reset_entries(self, frame):
		for e in self.entries:
			e[1].delete(0,'end')
	
	def callback(self, event):
		webbrowser.open_new(r"http://www.alarm.com/privacy_policy.aspx")
	
	def show_frame(self, page_name):
		'''Show a frame for the given page name'''
		frame = self.frames[page_name]
		frame.tkraise()
	
	def make_text(self, frame, text, font = ('Arial',20, 'bold')):
		lab = tk.Label(frame, fg='black', bg=self.bg_color, wraplength = self.text_wrap_l, font=font, text=text)
		lab.pack(side=TOP, padx=self.text_padx, fill=X, pady =self.pady)
	
	def display_empty_mesg (self, field):
		messagebox.showerror(field, field+' field cannot be empty!')
	
	def display_invalid_msg (self, field, text):
		messagebox.showerror(field, text+'is not a valid '+field+', Please retry with a valid '+field+'!')
	
	def byte_string_decoder(self, S):
		return S.decode('AStext_dbCII')
	
	def update_textdb(self, text_datafile):
		#self.temp_ar = loadtxt(text_datafile, dtype='S50', delimiter=',')
		#if not self.temp_ar.size:
		#    self.temp_ar = np.empty([1, len(self.temp_ar[0])], dtype='S50')
		#self.temp_ar = self.bsd(self.temp_ar)
		#temp_keys = self.temp_ar[:,0]
		#temp_values = self.temp_ar[:,1:]
		#self.text_db = dict(zip(temp_keys, temp_values))
		#self.text_db = temp_keys
		with open (self.text_datafile, 'rb') as fp:
			self.text_db = pickle.load(fp)
		return self.text_db
	
	def upload_video(self, username):
		output_db = os.path.join(os.getcwd(),'db')
		output_dir = os.path.join(output_db,username)
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		num = str(len(os.listdir(output_dir))+1)
		output_video_file = os.path.join(output_dir, '_'+num+'.avi')
		camera = webcam()
		camera.cap_video(length = 10, output_file = output_video_file, fps = 30.0, resolution = (640,480), prepare = True, prep_time = 5)
		confirm = messagebox.askquestion('Verify','Want to see your video clip before submitting?')
		if confirm=='yes':
			camera.disp_video(output_video_file)
			redo = messagebox.askquestion('Re-record','Want to record your video clip again before submitting?')
			if redo=='yes':
				camera.cap_video(length = 10, output_file = output_video_file, fps = 30.0, resolution = (640,480), prepare = True, prep_time = 5)
		messagebox.showinfo('video upload', 'Successful...!')    
			
class StartPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		
		self.controller.make_text(self, text="Alarm.com Employee Facial Recognition Research and Development Data Collection Program", font = controller.font_for_titles)

		button1 = tk.Button(self, bg=controller.bg_color, text="Sign In", font=controller.font_for_buttons,command=lambda: controller.show_frame("PageOne"))
		button2 = tk.Button(self, bg=controller.bg_color, text="Sign Up", font=controller.font_for_buttons,
							command=lambda: controller.show_frame("PageTwo"))
		button1.pack(padx=controller.button_padx, pady = controller.pady, fill=X)
		button2.pack(padx=controller.button_padx, pady = controller.pady, fill=X)
		
		self.grand_parent_name = parent.winfo_parent()
		b2 = Button(self,bg = controller.bg_color, text='Quit', fg='red', font=controller.font_for_buttons, command=parent._nametowidget(self.grand_parent_name).destroy)

		b2.pack(padx=controller.button_padx, pady=controller.pady)

class PageTwo(tk.Frame):
	'''Creates sign up page'''
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		
		
		
		label = tk.Label(self, text="Sign up", bg=controller.bg_color,  font=controller.font_for_titles)
		label.pack(side="top", fill="x", pady=controller.pady)
		
		
		self.controller.make_text(frame=self, text = "Alarm.com Employee Facial Recognition Consent for Research and Development (dated as of June 9, 2017)", font=controller.font_for_titles)
		self.controller.make_text(frame=self, text = "By clicking 'I Agree' or signing below, I consent to Alarm.com's collection, transmission, maintenance, processing, and use of my video images and biometric data (collectively, 'Facial Recognition Data') in order to enable the research and development of Alarm.com's facial recognition services, as described in this notice and in accordance with Alarm.com's Privacy Policy. I further acknowledge and agree that Alarm.com may share my Facial Recognition Data with its affiliates for use in accordance with this notice and that Alarm.com and such affiliates may retain my Facial Recognition Data. I acknowledge and agree that I am 13 years old or older, currently an employee of Alarm.com and not a resident of the State of Illinois.  Please read our Privacy Policy and the acknowledgement below, and click 'I Agree', or sign below, to consent.",font=controller.font_for_text)
		self.controller.make_text(frame=self, text = "I have reviewed this notice, the Alarm.com Privacy Policy and hereby consent to the collection, use and disclosure of my Facial Recognition Data in accordance with the Privacy Policy and this notice.", font = controller.font_for_text)
		
		
		row = Frame(self, bg=controller.bg_color)
		agree = IntVar()
		c = Checkbutton(row, bg= controller.bg_color, text="I agree and give consent", font=controller.font_for_text,variable=agree)
		row.pack(side=TOP, padx = controller.text_padx, pady=controller.pady)
		c.pack(side=LEFT)
		
		
		link = Label(row, text="Alarm.com Privacy Policy", fg="blue", bg=controller.bg_color, font = controller.font_for_text, cursor="hand2")
		link.pack(side = LEFT, padx = (int(controller.scr_w*0.0521),controller.text_padx))
		link.bind("<Button-1>", controller.callback)
		
		
		fields = ['ADC email']
		ents = controller.makeform(self, fields)
		
		self.bind('<Return>', (lambda event, e=ents: self.validate_and_submit(e, agree)))
		
		b1 = Button(self, bg = controller.bg_color, text='Submit', font=controller.font_for_buttons,
			  command=(lambda e=ents: self.validate_and_submit(e, agree)))
		b1.pack(side = TOP, padx=controller.button_padx, pady=controller.pady)
		self.grand_parent_name = parent.winfo_parent()
		b2 = Button(self,bg = controller.bg_color, text='Quit', fg='red', font=controller.font_for_buttons, command=parent._nametowidget(self.grand_parent_name).destroy)
		b2.pack(padx=controller.button_padx, pady=controller.pady)
		
		
		button = tk.Button(self, bg=controller.bg_color, text="Go to the home page", font=controller.font_for_buttons,
						   command=lambda: controller.show_frame("StartPage"))
		button.pack(padx=controller.button_padx, pady=controller.pady)
	
	def submit_new_entry_to_textdb(self, entries):
		#self.controller.temp_ar = np.vstack((self.controller.temp_ar,np.asarray(entries, dtype='S50')))
		#np.savetxt(self.controller.text_datafile, self.controller.temp_ar, fmt=('%s'), delimiter=',')
		#self.controller.update_textdb(self.controller.text_datafile)
		self.controller.text_db.append(entries[0])
		with open(self.controller.text_datafile, 'wb') as fp:
			pickle.dump(self.controller.text_db, fp)
	
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
			if(field == 'ADC email'):
				if (text in self.controller.text_db):
					flag = 0
					messagebox.showerror(field+' IndexError: pop from empty listerror!' , 'User: '+ text +' already exists! Please use sign in option if you have already signed up!')
			if (field == 'ADC email' and not re.match(r"[^@]+@[^@]+\.[^@]+", text)):
				print('%s: is not a valid email! Please enter a valid email id!' % (text))
				flag=0
				self.controller.display_invalid_msg(field, text)
			if (field == 'ADC email'):
				domain = text.split('@')[1]
				valid_domains = ['alarm.com','Alarm.com','objectvideo.com','pointcentral.com','energyhub.net','securitytrax.com','building36.com']
				if domain not in valid_domains:
					mesg = str('%s: is not a valid ADC domain! Please enter a valid ADC employee email id!' % (domain))
					#print(mesg)
					flag=0
					#self.controller.display_invalid_msg(field, text)
					messagebox.showerror('Invalid ADC domain!',mesg)
			#^[a-zA-Z]+\s[a-zA-Z]+$
			#elif (field == 'Password' and not re.match(r"^[a-zA-Z]+\s[a-zA-Z]+$", text)):
			#    print('The Passphrase entered is not a valid one! Please enter two english words separated with only one space')
			#    flag=0
			#    self.controller.display_invalid_msg(field, 'entered passphrase')
		if flag and agree.get():
			self.submit_new_entry_to_textdb(text_entry_list)
			messagebox.showinfo('Registration status', 'Successful...!\nPlease read the following instructions carefully and follow them!\n1. Now your face video will be recorded for 10 seconds after 5 seconds of preparation time.\n2. Please adjust your face relative to the position of the webcamera so that your face is in the center.\n')
			self.controller.upload_video(text_entry_list[0])
			self.controller.reset_entries(self)
			self.controller.show_frame("StartPage")
			#print("Agreed and submitted")
			#self.contorller.root.destroy()
		elif (flag and not agree.get()):
			messagebox.showerror('Agreement required', 'You have to agree to the terms to proceed!\n Please agree after reviewing the terms...')
			#print("Not agreed and submitted!")
		else:
			#print('flag is reset for some reason!')
			pass

if __name__ == "__main__":
	app = SampleApp('db_users.pkl')
	app.mainloop()
