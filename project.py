
import cv2, os, pathlib, csv
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import urllib.request

BASE_DIR = pathlib.Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
DB_FILE = BASE_DIR / "students.csv"
REPORT_DIR = BASE_DIR / "reports"
CASCADE_LOCAL = BASE_DIR / "haarcascade.xml"
MODEL_PATH = BASE_DIR / "trainer.yml"
NAMES_FILE = BASE_DIR / "names.txt"

DATASET_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

if not DB_FILE.exists():
    with open(DB_FILE,'w', newline='') as f:
        csv.writer(f).writerow(["id","name","roll","class_name"])

if not CASCADE_LOCAL.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", str(CASCADE_LOCAL))

def get_cascade(): return cv2.CascadeClassifier(str(CASCADE_LOCAL))

def get_names():
    names=[]
    if NAMES_FILE.exists():
        for line in open(NAMES_FILE, encoding='utf-8'):
            if ',' in line:
                names.append(line.strip().split(',',1)[1])
    return names

def parse_folder(folder_id):
    try:
        parts = folder_id.split('_')
        if len(parts) >= 3:
            roll_raw = parts[1]
            name_raw = " ".join(parts[2:]).replace("_"," ").strip()
            roll_clean = roll_raw.upper().replace("MCA-","").replace("MCA","").strip("-")
            if roll_clean=="": roll_clean="574"
            return name_raw, f"MCA-{roll_clean}"
        else:
            return folder_id, "MCA-574"
    except:
        return folder_id, "MCA-574"

def get_session():
    hour = datetime.now().hour
    return "AM" if hour < 12 else "PM"

def train_model():
    faces=[]; ids=[]; names=[]; mapping={}; cur=0
    for person in DATASET_DIR.iterdir():
        if not person.is_dir(): continue
        if person.name not in mapping:
            mapping[person.name]=cur; names.append(person.name); cur+=1
        for img in person.glob("*.jpg"):
            g=cv2.imread(str(img),0)
            if g is None: continue
            faces.append(cv2.resize(g,(200,200)))
            ids.append(mapping[person.name])
    if not faces: return None, None
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, __import__('numpy').array(ids))
    rec.write(str(MODEL_PATH))
    with open(NAMES_FILE,'w', encoding='utf-8') as f:
        for i,n in enumerate(names):
            f.write(f"{i},{n}\n")
    return rec, names

def register_student():
    name=name_var.get().strip()
    roll=roll_var.get().strip()
    class_name=class_var.get().strip()
    if not name or not roll:
        messagebox.showerror("Error","Fill Name and Roll")
        return
    import shutil
    if DATASET_DIR.exists():
        for p in DATASET_DIR.iterdir():
            if p.is_dir(): shutil.rmtree(p)
    for fp in [MODEL_PATH, NAMES_FILE, DB_FILE]:
        if pathlib.Path(fp).exists(): os.remove(fp)
    with open(DB_FILE,'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["id","name","roll","class_name"])
    full_id = f"{class_name}_{roll}_{name}".replace(" ","_")
    save_path=DATASET_DIR / full_id
    save_path.mkdir(exist_ok=True)
    with open(DB_FILE,'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([full_id,name,roll,class_name])
    root.iconify()
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    face_cascade=get_cascade()
    count=0
    while True:
        ret,frame=cap.read()
        if not ret: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        for (x,y,w,h) in face_cascade.detectMultiScale(gray,1.3,5):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{count}/20 Press s",(10,30),1,2,(0,255,0),2)
        cv2.imshow("Register - s to save, q to quit",frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('s'):
            faces=face_cascade.detectMultiScale(gray,1.3,5)
            if len(faces)==0: continue
            for (x,y,w,h) in faces:
                cv2.imwrite(str(save_path / f"{count}.jpg"), cv2.resize(gray[y:y+h,x:x+w],(200,200)))
                count+=1
                break
        if key==ord('q') or count>=20:
            break
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()
    root.lift()
    if count>0:
        train_model()
        messagebox.showinfo("Done",f"Registered {name} as MCA-574")

def take_attendance():
    if not MODEL_PATH.exists():
        messagebox.showerror("Error","Register first!")
        return
    session = get_session()
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.read(str(MODEL_PATH))
    names=get_names()
    face_cascade=get_cascade()
    root.iconify()
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    marked=set()
    data=[]
    while True:
        ret,frame=cap.read()
        if not ret: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            roi=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            try:
                id_,conf=rec.predict(roi)
            except:
                continue
            if conf<70 and id_<len(names):
                folder_id=names[id_]
                real_name, roll_display = parse_folder(folder_id)
                # VERIFICATION WINDOW: ONLY NAME AND ROLL
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"{real_name}",(x,y-25),1,1.2,(0,255,0),2)
                cv2.putText(frame,f"{roll_display}",(x,y-8),1,1.2,(0,255,0),2)
                if folder_id not in marked:
                    now=datetime.now()
                    data.append([real_name, roll_display, "MCA", now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), session])
                    marked.add(folder_id)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,"Unknown",(x,y-10),1,1.2,(0,0,255),2)
        cv2.imshow("Attendance - Press q to save",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()
    root.lift()
    if data:
        df=pd.DataFrame(data,columns=["Name","Roll","Class","Date","Time","Session"])
        df = df.drop_duplicates(subset=["Name","Roll","Session"])
        file=REPORT_DIR / f"attendance_{datetime.now().strftime('%Y-%m-%d')}_{session}.xlsx"
        if file.exists():
            old=pd.read_excel(file)
            df=pd.concat([old,df]).drop_duplicates(subset=["Name","Roll"])
        df.to_excel(file,index=False)
        monthly=REPORT_DIR / f"monthly_{datetime.now().strftime('%Y-%m')}.xlsx"
        if monthly.exists():
            old=pd.read_excel(monthly)
            df_all=pd.concat([old,df]).drop_duplicates(subset=["Name","Date","Session"])
            df_all.to_excel(monthly,index=False)
        else:
            df.to_excel(monthly,index=False)
        messagebox.showinfo("Saved",f"{session} Saved!\n{file.name}")

def view_reports():
    REPORT_DIR.mkdir(exist_ok=True)
    os.startfile(str(REPORT_DIR))

root=tk.Tk()
root.title("SMART ATTENDANCE SYSTEM")
root.geometry("520x450")
root.configure(bg="#eef2ff")
tk.Label(root,text="SMART ATTENDANCE SYSTEM",font=("Arial",16,"bold"),bg="#eef2ff").pack(pady=20)
frame=tk.Frame(root,bg="#eef2ff")
frame.pack(pady=10)
tk.Label(frame,text="Student Name:",bg="#eef2ff").grid(row=0,column=0,sticky='w',pady=6,padx=5)
name_var=tk.StringVar()
tk.Entry(frame,textvariable=name_var,width=28).grid(row=0,column=1,pady=6)
tk.Label(frame,text="Roll No:",bg="#eef2ff").grid(row=1,column=0,sticky='w',pady=6,padx=5)
roll_var=tk.StringVar()
tk.Entry(frame,textvariable=roll_var,width=28).grid(row=1,column=1,pady=6)
tk.Label(frame,text="Class/Course:",bg="#eef2ff").grid(row=2,column=0,sticky='w',pady=6,padx=5)
class_var=tk.StringVar()
tk.Entry(frame,textvariable=class_var,width=28).grid(row=2,column=1,pady=6)
class_var.set("MCA")
tk.Button(root,text="1. REGISTER STUDENT",bg="#4CAF50",fg="white",font=("Arial",11,"bold"),width=38,height=2,command=register_student).pack(pady=8)
tk.Button(root,text="2. TAKE ATTENDANCE",bg="#2196F3",fg="white",font=("Arial",11,"bold"),width=38,height=2,command=take_attendance).pack(pady=8)
tk.Button(root,text="3. VIEW REPORTS",bg="#FF9800",fg="white",font=("Arial",11,"bold"),width=38,height=2,command=view_reports).pack(pady=8)
root.mainloop()
