# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:52:02 2023

@author: shank
"""

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from tkcalendar import Calendar
import pandas as pd
from datetime import datetime, timezone
import requests
import time
import webbrowser
import numpy as np
import pytz
from PIL import Image, ImageTk
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

__version__ = "v2.1.0"
__date__ = "5th July 2024"
__auth__ = "pk_3326657_EOM3G6Z3CKH2W61H8NOL5T7AGO9D7LNN"
# Dictionary mapping month names to numbers
month_dict = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7,
    "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

# Define the list of columns to check for NaN
columns_to_check = [
    "Course", "Product", "Proj-Common-Activity", "Proj-Outside-Office",
    "Management-Project", "Technology-Project", "Linguistic-Project",
    "MMedia-Project", "Project-CST", "Sales-Mktg-Project", "Project-ELA",
    "Proj-KidsPersona", "FinAcc-Project", "Website", "SFH-Admin-Project", 
    "Admin-Project", "Linguistic-Activity"
]

# Create a timezone object for IST
ist_timezone = pytz.timezone('Asia/Kolkata')

# Exchange keys and values
month_flipped = {value: key for key, value in month_dict.items()}

def download_latest(root):
        
    # Define the GitHub repository, file path, and file URL
    repository = "Vyoma-Linguistic-Labs/ClickUp_Timesheets"
    file_path = "Generate_Timesheet.py"  # Replace with the path to the file you want to download
    file_url = f"https://raw.githubusercontent.com/{repository}/main/{file_path}"

    # Define the local file name where you want to save the downloaded file
    local_file_name = "Generate_Timesheet.py"

    try:
        # Send an HTTP GET request to the file URL
        response = requests.get(file_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the file content to a local file
            with open(local_file_name, "wb") as file:
                file.write(response.content)
            print(f"File '{local_file_name}' downloaded successfully.")
            message = "The latest version of Generate_Timesheet.py is downloaded."\
                "\nClose this window and re-run the script."
            messagebox.showinfo("Success!", message)
            root.destroy()
            import sys
            sys.exit()
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            messagebox.showerror("Error!", 
                                 f"Failed to download file. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {str(e)}")
        messagebox.showerror("Error!", f"An error occurred: {str(e)}")
    return

def check_for_update(current_version):
    url = "https://api.github.com/repos/Vyoma-Linguistic-Labs/ClickUp_Timesheets/releases/latest"
    response = requests.get(url)
    if response.status_code == 200:
        latest_version = response.json()["tag_name"]
        if latest_version != current_version:
            print(f"A new version ({latest_version}) is available. Please download it from GitHub.")
            root = tk.Tk()
            root.title("Newer Version available")
            
            message = f"A new version ({latest_version}) is available."\
                "\nClick on the button below to automatically download the"\
                    " \nlatest version from the GitHub repository."
            label = tk.Label(root, text=message, #fg="blue", cursor="hand2",
                             font=("Times", 12, "bold"))
            label.pack()        
            
            # Submit Button
            submit_button = tk.Button(root, text=f"Download Latest Version {latest_version}",
                                      command=lambda arg=root: download_latest(arg))
            submit_button.pack(pady=10)
                        
            root.mainloop()
            import sys
            sys.exit()

def convert_milliseconds_to_hours_minutes(milliseconds):
    seconds = milliseconds / 1000
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    return (int(hours), int(minutes))

def memberInfo():
    url = "https://api.clickup.com/api/v2/team"
    headers = {"Authorization": __auth__}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Extract member id and username
    members_dict = {}
    for team in data['teams']:
        for member in team['members']:
            member_id = member['user']['id']
            member_username = member['user']['username']
            members_dict[member_id] = member_username

    # Exchange keys and values - keep last 4 digits corresponding to emp ID
    members_dict = {value[-4:]: key for key, value in members_dict.items() if value is not None}
        
    return members_dict

def open_link(link):    
    webbrowser.open_new("app.clickup.com/t/"+link)
        
def show_progress_window(title, message):
    """Create and return a progress window with a progress bar."""
    prog_win = tk.Toplevel(root)
    prog_win.title(title)
    prog_win.geometry("420x120")
    prog_win.resizable(False, False)
    # Center over root
    root.update_idletasks()
    px = root.winfo_rootx() + root.winfo_width() // 2 - 210
    py = root.winfo_rooty() + root.winfo_height() // 2 - 60
    prog_win.geometry(f"+{px}+{py}")
    prog_win.grab_set()  # Modal

    lbl = tk.Label(prog_win, text=message, font=("Arial", 11), wraplength=380)
    lbl.pack(pady=(15, 5))

    bar = ttk.Progressbar(prog_win, mode="indeterminate", length=360)
    bar.pack(pady=5)
    bar.start(12)

    status_lbl = tk.Label(prog_win, text="", font=("Arial", 9), fg="grey")
    status_lbl.pack()

    prog_win.update()
    return prog_win, bar, lbl, status_lbl


def fetch_task_details(task_id, headers):
    """Fetch details for a single task from ClickUp API."""
    url = "https://api.clickup.com/api/v2/task/" + task_id
    try:
        response = requests.get(url, headers=headers, timeout=30)
        return task_id, response.json()
    except Exception as e:
        print(f"Error fetching task {task_id}: {e}")
        return task_id, None


def get_selected_dates():
    start_date = start_cal.selection_get()
    end_date = end_cal.selection_get()
    key = key_entry.get().upper()

    # Format dates
    start_date_str = start_date.strftime("%b %d")
    end_date_str = end_date.strftime("%b %d")
    year_str = str(start_date.year)

    # Generate filename
    filename = f"{key}_{start_date_str}_to_{end_date_str}_{year_str}.xlsx"
    
    # Retrieve information from ClickUp
    start = time.time()

    # Show progress window while fetching member info
    prog_win, bar, lbl, status_lbl = show_progress_window(
        "Fetching Data", "Step 1/3: Fetching member info from ClickUp...")
    root.update()

    members_dict = memberInfo()
    employee_key = members_dict[key] # Convert our key to ClickUp key
       
    date_obj = datetime.strptime(str(start_date), '%Y-%m-%d')
    start_timestamp = int(date_obj.replace(tzinfo=timezone.utc).timestamp())
    date_obj = datetime.strptime(str(end_date), '%Y-%m-%d')
    end_timestamp = int(date_obj.replace(tzinfo=timezone.utc).timestamp())

    team_id = "3314662"
    url = "https://api.clickup.com/api/v2/team/" + team_id + "/time_entries"
    query = {
      "start_date": str(int(start_timestamp - 19800)*1000),
      "end_date": str(int((end_timestamp+86399)*1000) - 19800000),
      "assignee": employee_key,
    }
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": __auth__
    }

    lbl.config(text="Step 2/3: Fetching time entries from ClickUp...")
    prog_win.update()
    
    response = requests.get(url, headers=headers, params=query)
    data = response.json()    
    
    # Initialize empty lists for each column
    task_names = []
    task_ids = []
    task_status = []
    durations = []
    dates = []
    days = []
    
    # Loop through the data and extract the required fields
    for entry in data['data']:
        try:
            task_names.append(entry['task']['name'])
            task_ids.append(entry['task']['id'])
            task_status.append(entry['task']['status']['status'])
        except:
            task_names.append('0')
            task_ids.append('0')
            task_status.append('0')
        durations.append(int(entry['duration']))
        start_time = int(entry['start']) // 1000 # Convert to seconds
        
        date = pd.Timestamp(start_time, unit='s').date()
        dates.append(date)

        # Convert start_time to a datetime object in UTC, Localize the datetime object to UTC                
        localized_start_datetime = pytz.utc.localize(datetime.utcfromtimestamp(start_time))
        # Convert the datetime object from UTC to IST
        day = localized_start_datetime.astimezone(ist_timezone).strftime('%A')
        days.append(day)
        
    # Create a pandas dataframe
    df = pd.DataFrame({
        'Task Name': task_names,
        'Task ID': task_ids,
        'Task Status': task_status,
        'Duration': durations,
        'Date': dates,
        'Day': days
    })
    
    # Create a new DataFrame with only unique Task IDs
    task_ids = df['Task ID'].unique()
    new_df = pd.DataFrame({'Task ID': task_ids})
    
    # Add columns for each day of the week
    days_of_week = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 
                    'Thursday', 'Friday']
    for day in days_of_week:
        new_df[day] = 0
    
    # Loop through each task and add duration to the corresponding day column
    for task in task_ids:
        task_entries = df[df['Task ID'] == task]
        grouped_entries = task_entries.groupby(['Day']).sum(numeric_only=True)
        for day in days_of_week:
            if day in grouped_entries.index:
                new_df.loc[new_df['Task ID'] == task, day] = grouped_entries.loc[day]['Duration']
    
    # Merge the new DataFrame with the original DataFrame
    df_h = pd.merge(df, new_df, on='Task ID')
    
    # Drop duplicates
    df_h.drop_duplicates(subset='Task ID', inplace=True)
    
    # Convert the durations to hours format
    df_h[days_of_week] = df_h[days_of_week].apply(lambda x: x / 3600000).round(2)
    df_h = df_h.drop(['Duration', 'Date', 'Day'], axis=1)
    
    # --- PARALLEL fetch of individual task details ---
    api_headers = {"Authorization": __auth__}

    # Skip invalid '0' task IDs — they have no data in ClickUp
    unique_task_ids = [tid for tid in df_h['Task ID'].unique() if tid != '0']
    total_tasks = len(unique_task_ids)

    lbl.config(text=f"Step 3/3: Fetching details for {total_tasks} tasks (parallel)...")
    bar.stop()
    bar.config(mode="determinate", maximum=max(total_tasks, 1), value=0)
    prog_win.update()

    completed = [0]

    MAX_WORKERS = 50       # increased from 20
    MAX_RETRIES = 2        # retry failed requests once
    task_results = {}

    def fetch_with_retry(task_id, headers):
        """Fetch task details with retry on failure."""
        for attempt in range(MAX_RETRIES + 1):
            tid, result = fetch_task_details(task_id, headers)
            if result is not None:
                return tid, result
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))  # brief back-off before retry
        return task_id, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_with_retry, tid, api_headers): tid
                   for tid in unique_task_ids}
        for future in as_completed(futures):
            task_id, result = future.result()
            task_results[task_id] = result
            completed[0] += 1
            bar["value"] = completed[0]
            status_lbl.config(text=f"{completed[0]} / {total_tasks} tasks fetched")
            prog_win.update()

    # Apply results to dataframe
    for task_id, tasks in task_results.items():
        if tasks is None:
            continue
        try:
            hrs_mins = convert_milliseconds_to_hours_minutes(tasks['time_spent'])
            df_h.loc[df_h['Task ID'] == task_id,
                     'Total Time tracked for this task till now (hrs)'] = (
                         str(hrs_mins[0]) + 'h ' + str(hrs_mins[1]) + 'm')
        except Exception:
            pass
        try:
            for custom_field in tasks['custom_fields']:
                if 'value' in custom_field:
                    if custom_field['type'] == 'drop_down':
                        df_h.loc[df_h['Task ID'] == task_id, custom_field['name']] = (
                            custom_field['type_config']['options'][custom_field['value']]['name'])
        except Exception:
            pass

    prog_win.destroy()  # Close progress window
    
    # Check if 'Proj-Common-Activity' column exists in the DataFrame
    if 'Proj-Common-Activity' in df_h.columns:
        # Filter out rows where 'Proj-Common-Activity' is 'Vyoma Holiday' or 'Personal Leave'
        df_h = df_h[(df_h['Proj-Common-Activity'] != 'Vyoma Holiday') & (df_h['Proj-Common-Activity'] != 'Personal Leave')]
    
    # Check if 'Goal Type' column exists
    if 'Goal Type' not in df_h.columns:
        # Add a new column with 'nan' values
        df_h['Goal Type'] = np.nan
    # Initialize a list to collect the names of rows that do not fit the criterion
    rows_with_missing_data = []
    row_id_with_missing_data = []
    rows_missing_goal_type = []
    row_id_missing_goal_type = []
    project_columns = list(set(df_h.columns.tolist()).intersection(columns_to_check))
    # Iterate through rows in the DataFrame
    for index, row in df_h.iterrows():
        # Extract the 'Task Name' column value for the current row
        task_name = row['Task Name']
        task_id = row['Task ID']        
            
        if all(row[col] == 'nan' for col in project_columns):  # All specified columns are "nan" (string literal)
            rows_with_missing_data.append(task_name)
            row_id_with_missing_data.append(task_id)
        
        if row['Goal Type'] == 'nan':  # Goal Type is "nan" (string literal)
            rows_missing_goal_type.append(task_name)
            row_id_missing_goal_type.append(task_id)

            
    # Output the names of rows that do not fit the criterion
    if rows_with_missing_data or rows_missing_goal_type:
        # Create a new top-level window for the error message
        error_window = tk.Toplevel(root)
        error_window.title("ERROR")
        # second_win = tkinter.Toplevel(root)
        root.eval(f'tk::PlaceWindow {str(error_window)} center')
        
        if rows_with_missing_data:
            # Add widgets to the error window to display the error message
            error_label = tk.Label(error_window, 
                                   text="‘Project/Product/Course/Website’ is not"\
                                       " set for the below task(s) (links provided)",
                                   font=("Times", 12, "bold"))
            error_label.pack()
                  
            # Create labels for each link
            link_label = []
            count = 0
            for link_text, link_url in zip(rows_with_missing_data, row_id_with_missing_data):
                            
                label = tk.Label(error_window, text=link_text, fg="blue", cursor="hand2")
                label.pack()        
                # Bind the label to the open_link function with the corresponding link_url
                label.bind("<Button-1>", lambda e, url=link_url: open_link(url))
                link_label.append(label)
                
                count += 1
        # You can add more widgets here to provide additional information
        if rows_missing_goal_type:
            # Add widgets to the error window to display the error message
            goal_error_label = tk.Label(error_window, 
                                   text="Goal Type not set for (links provided):",
                                   font=("Times", 12, "bold"))
            goal_error_label.pack()
                  
            # Create labels for each link
            goal_link_label = []
            count = 0
            for link_text, link_url in zip(rows_missing_goal_type, row_id_missing_goal_type):
                            
                label = tk.Label(error_window, text=link_text, fg="blue", cursor="hand2")
                label.pack()        
                # Bind the label to the open_link function with the corresponding link_url
                label.bind("<Button-1>", lambda e, url=link_url: open_link(url))
                goal_link_label.append(label)
                
                count += 1
        
        info_label = tk.Label(error_window, 
                               text="You can try generating your timesheet again once"\
                                   " you set the above information in these tasks.",
                               font=("Times", 12, "bold"))
        info_label.pack()
        # Start the mainloop for the error window
        error_window.mainloop()    
    
    # Add the 'time_this_week' column by summing the values of all days_of_week columns
    df_h['Total Tracked this week in this task'] = df_h[days_of_week].sum(axis=1)
    # Calculate the totals of the days_of_week columns
    totals = df_h[days_of_week].sum(axis=0)

    # Append totals as a new row to the DataFrame
    df_h = pd.concat([df_h, totals.to_frame().T], ignore_index=True)
    # Sum the values in the last row of the DataFrame
    weekly_total = df_h.iloc[-1].sum()
    
    if  pd.isna(df_h.at[df_h.index[0], 'Task Status']):
        # Create a new top-level window for the error message
        error_window = tk.Toplevel(root)
        error_window.title("ERROR")
    
        # Set the size of the error window
        window_width, window_height = 400, 200
        error_window.geometry(f"{window_width}x{window_height}")
    
        # Calculate the position to center the error window with respect to the root window
        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        root_width = root.winfo_width()
        root_height = root.winfo_height()
    
        position_right = root_x + int(root_width / 2) - int(window_width / 2)
        position_down = root_y + int(root_height / 2) - int(window_height / 2)
    
        error_window.geometry(f"+{position_right}+{position_down}")
    
        # Add padding around the message
        padding = {"padx": 20, "pady": 20}
    
        # Add widgets to the error window to display the error message
        error_label = tk.Label(
            error_window,
            text="There are no entries in this Date Range."
                 "\n\nPlease change Date Range or Update Entries in ClickUp",
            font=("Arial", 16, "bold"),
            wraplength=360,  # Wrap text within 360 pixels
            **padding
        )
        error_label.pack(expand=True)
    
        error_window.mainloop()  
    # Update the value in the 'Status' column for the last row 
    df_h.at[df_h.index[-1], 'Task Status'] = 'Daily Totals ->'
    
    # Create an empty row with NaN values
    empty_row = pd.Series([np.nan] * len(df_h.columns), index=df_h.columns)
    # Append the empty row to the DataFrame
    df_h = pd.concat([df_h, empty_row.to_frame().T], ignore_index=True)    
    # Append a value to the 6th column
    df_h.iloc[-1, 5] = weekly_total
    if int((end_timestamp - start_timestamp)/86400)+1 <= 7: 
        df_h[df_h.columns[3]] = df_h[df_h.columns[3]].astype("object")  # Convert the entire column to object type           
        df_h.iloc[-1, 3] = 'Week\'s total ='
        week_number = end_date.isocalendar()[1]
        df_h.at[df_h.index[-1], 'Task Name'] = f'Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}'
    else:
        df_h[df_h.columns[3]] = df_h[df_h.columns[3]].astype(object)
        df_h.iloc[-1, 3] = 'Total Hours Tracked ='
        df_h.at[df_h.index[-1], 'Task Name'] = f'{start_date_str}, {year_str} - {end_date_str}, {year_str}'
    
    # Move the column to the 11th position
    df_h.insert(10, 'Total Tracked this week in this task', 
                df_h.pop('Total Tracked this week in this task'))

    # Helper to write dataframe to an excel path with hyperlinks
    def write_excel(path, df):
        w = pd.ExcelWriter(path, engine='xlsxwriter')
        df.to_excel(w, sheet_name='Sheet1', index=False)
        ws = w.sheets['Sheet1']
        for row_num, value in enumerate(df['Task ID'], start=1):
            if pd.isna(value):
                break
            ws.write_url(row_num, df.columns.get_loc('Task ID'),
                         f'https://app.clickup.com/t/{value}', string=value)
        w.close()

    # Always save a copy in the script's own folder first
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, filename)
    write_excel(local_path, df_h)

    # Also open Save As dialog so user can save a copy anywhere they like
    save_path = filedialog.asksaveasfilename(
        title="Save Timesheet As (optional – already saved in script folder)",
        initialdir=os.path.expanduser("~\\Desktop"),
        initialfile=filename,
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if save_path and save_path != local_path:
        write_excel(save_path, df_h)
    else:
        save_path = local_path

    # Update output label with saved path
    output_label.config(
        text=f"✅ Timesheet saved:\n{save_path}"
    )

    # Show a download/open confirmation popup with an Open File button
    def open_saved_file():
        import os
        os.startfile(save_path)

    def open_saved_folder():
        import os, subprocess
        subprocess.Popen(f'explorer /select,"{save_path}"')

    confirm_win = tk.Toplevel(root)
    confirm_win.title("Timesheet Ready")
    confirm_win.geometry("460x160")
    confirm_win.resizable(False, False)
    root.update_idletasks()
    cx = root.winfo_rootx() + root.winfo_width() // 2 - 230
    cy = root.winfo_rooty() + root.winfo_height() // 2 - 80
    confirm_win.geometry(f"+{cx}+{cy}")

    tk.Label(confirm_win, text="✅ Timesheet generated successfully!",
             font=("Arial", 12, "bold"), fg="green").pack(pady=(15, 5))
    tk.Label(confirm_win, text=save_path, font=("Arial", 9), fg="grey",
             wraplength=440).pack(pady=(0, 10))

    btn_frame = tk.Frame(confirm_win)
    btn_frame.pack()
    tk.Button(btn_frame, text="📂 Open File", width=16, bg="#4CAF50", fg="white",
              font=("Arial", 10, "bold"),
              command=lambda: (open_saved_file(), confirm_win.destroy())).pack(side="left", padx=8)
    tk.Button(btn_frame, text="📁 Show in Folder", width=16, bg="#2196F3", fg="white",
              font=("Arial", 10),
              command=lambda: (open_saved_folder(), confirm_win.destroy())).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Close", width=10,
              command=confirm_win.destroy).pack(side="left", padx=8)

    total_time = df_h['Total Tracked this week in this task'].sum()
    print("Total Hours for this time frame: ", total_time)
    print(f"Saved to: {save_path}")
    print(time.time()-start)

    # Clear the selected dates and employee key
    start_cal.selection_clear()
    end_cal.selection_clear()
    # key_entry.delete(0, tk.END)
    if checkbox_var_1.get():
        # URLs to open
        url1 = 'https://docs.google.com/spreadsheets/d/1XLDSTT5m952eiOXhiUtxldIIoEAfQgiVKv5XY2HFOBg/edit?usp=sharing'
        # Open URLs in separate browser tabs
        # time.sleep(5)
        webbrowser.open_new_tab(url1)
    
    ### Draft mail Feature - for later version
    # if checkbox_var_2.get():
    #     time.sleep(5)            
    #     recipient = "venkat.s@vyomalabs.in"
    #     recipient_cc = "srilatha.vyoma@gmail.com"
    #     subject = f"Timesheet for Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}"
    #     body = "Ram ram ram, \nPlease find my weekly timesheet in this Google tracker -"
    #     # Encode special characters in the subject and body
    #     subject = urllib.parse.quote(subject)
    #     body = urllib.parse.quote(body)
    #     # Compose the Gmail URL with recipient, subject, and body
    #     gmail_url = f"https://mail.google.com/mail/?view=cm&to={recipient}&cc={recipient_cc}&su={subject}&body={body}"    
    #     # Open the Gmail URL in a web browser
    #     webbrowser.open(gmail_url)
    # root.destroy()
if __name__ == "__main__":
    current_version = __version__  # Replace with your current version
    check_for_update(current_version)
    # Rest of your script
    
    root = tk.Tk()
    root.title("Timesheet Generator")
    
    # Increase the size of the window
    window_width = 900
    window_height = 750
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    # Change the background color of the window
    root.configure(background="lightblue")
    
    try:
        # URL of the image you want to display
        image_url = "https://digitalsanskritguru.com/wp-content/uploads/" \
                    "2020/05/Vyoma_Logo_Blue_500x243.png"
        # Fetch the image from the URL
        response = requests.get(image_url)
        image_data = response.content
        # Create a PIL Image object from the image data
        image = Image.open(BytesIO(image_data))
        # Resize the image to the desired size
        image = image.resize((167, 81), Image.LANCZOS)
        # Convert the PIL Image to a PhotoImage object
        photo = ImageTk.PhotoImage(image)
        # Create a label to display the image
        image_label = tk.Label(root, image=photo)
        image_label.photo = photo 
        # Position the label at the top right corner
        image_label.pack(anchor=tk.NE, padx=5, pady=5)
    except:
        pass
    
    # Start Date Frame
    start_frame = tk.Frame(root)
    start_frame.pack(anchor=tk.CENTER)
    
    # Start Date Label
    start_label = tk.Label(start_frame, text="Start Date:", bg="black", fg="white")
    start_label.pack(side="left")
    
    # Start Date Calendar
    start_cal = Calendar(start_frame, selectmode="day", date_pattern="yyyy-mm-dd")
    start_cal.pack(side="left")
    
    # End Date Frame
    end_frame = tk.Frame(root)
    end_frame.pack(pady=10)
    
    # End Date Label
    end_label = tk.Label(end_frame, text="End Date:", bg="black", fg="white")
    end_label.pack(side="left")
    
    # End Date Calendar
    end_cal = Calendar(end_frame, selectmode="day", date_pattern="yyyy-mm-dd")
    end_cal.pack(side="left")
    
    # Employee Key Label
    key_label = tk.Label(root, text="Employee ID: (Eg: C047)", bg="black", fg="white")
    key_label.pack(pady=10)
    
    # Employee Key Entry
    key_entry = tk.Entry(root)
    key_entry.pack()
    
    # Create a variable to store the checkbox state
    checkbox_var_1 = tk.IntVar()
    # Create the checkbox
    checkbox_1 = tk.Checkbutton(root, text="Open the Google sheet for TS Submission Status", 
                                variable=checkbox_var_1)
    checkbox_1.pack()
    
    # # Create a variable to store the checkbox state
    # checkbox_var_2 = tk.IntVar()
    # # Create the checkbox
    # checkbox_2 = tk.Checkbutton(root, text="Open draft mail to send TS", 
    #                             variable=checkbox_var_2)
    # checkbox_2.pack()
    
    # Output Label
    output_label = tk.Label(root, 
                            text="Please update the above tracker once you send the Timesheet mail.")
                            # font=font.Font(weight="bold"))
    output_label.pack()
    
    # Submit Button
    submit_button = tk.Button(root, text="Submit", command=get_selected_dates)
    submit_button.pack(pady=10)
    
    # Output Label
    output_label = tk.Label(root, 
                            text="Note: Please find the generated Excel output in this folder itself.")
                            # font=font.Font(weight="bold"))
    output_label.pack()
    
    # Create the footer label
    footer_label = tk.Label(root, text="Version " + __version__ +" "+__date__, 
                            relief=tk.RAISED, anchor=tk.W)
    footer_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    root.mainloop()
