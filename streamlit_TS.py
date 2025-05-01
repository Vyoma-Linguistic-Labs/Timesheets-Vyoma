# timesheet_generator_streamlit.py

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:52:02 2023

Converted to Streamlit app on Fri, 4th Oct 2024.

@author: shank
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import time
import numpy as np
import pytz
from PIL import Image
from io import BytesIO
import urllib.parse
import webbrowser
import json

#api_key = st.secrets["auth"]
#team_id = st.secrets["team_id"]
api_key = "pk_3326657_EOM3G6Z3CKH2W61H8NOL5T7AGO9D7LNN"
team_id = "3314662"

__version__ = "v4.0.0"
__date__ = "1st May 2025"
__auth__ = api_key

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
    "Proj-KidsPersona", "Project-Finance", "Website", "SFH-Admin-Project",
    "Admin-Project", "Linguistic-Activity", "HR & Admin"
]

# Create a timezone object for IST
ist_timezone = pytz.timezone('Asia/Kolkata')

# Check if value is string 'nan' or np.nan
def is_nan(value):
    return value == 'nan' or (isinstance(value, float) and np.isnan(value))

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

# Optimized `get_selected_dates` function
def get_selected_dates(start_date, end_date, key, open_google_sheet, to_email, cc_email):
    if not key:
        st.error("Please enter your Employee ID.")
        return

    key = key.upper()
        
    # Format dates
    start_date_str = start_date.strftime("%b %d")
    end_date_str = end_date.strftime("%b %d")
    year_str = str(start_date.year)
    
    st.session_state['start_date_str'] = start_date_str
    st.session_state['end_date_str'] = end_date_str
    st.session_state['year_str'] = year_str
    
    # Generate filename
    filename = f"{key}_{start_date_str}_to_{end_date_str}_{year_str}.xlsx"

    # Retrieve information from ClickUp
    start_time_process = time.time()

    members_dict = memberInfo()
    if key not in members_dict:
        st.error("Invalid Employee ID. Please check and try again.")
        return

    employee_key = members_dict[key]

    # Convert start_date and end_date to timestamps
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())

    url = f"https://api.clickup.com/api/v2/team/{team_id}/time_entries"
    query = {
        "start_date": str((start_timestamp - 19800) * 1000),  # Convert to milliseconds
        "end_date": str((end_timestamp + 86399) * 1000 - 19800000),
        "assignee": employee_key,
    }

    headers = {"Content-Type": "application/json", "Authorization": __auth__}
    response = requests.get(url, headers=headers, params=query)
    data = response.json()

    if 'data' not in data or not data['data']:
        st.error("No entries found in this date range. Please update entries in ClickUp.")
        return

    # Extract task data efficiently
    task_data = [
        {
            "Task Name": entry.get('task', {}).get('name', '0'),
            "Task ID": entry.get('task', {}).get('id', '0'),
            "Task Status": entry.get('task', {}).get('status', {}).get('status', '0'),
            "Duration": int(entry['duration']),
            "Date": pd.Timestamp(int(entry['start']) // 1000, unit='s').date(),
            "Day": pytz.utc.localize(datetime.utcfromtimestamp(int(entry['start']) // 1000))
            .astimezone(ist_timezone)
            .strftime('%A'),
        }
        for entry in data['data']
    ]

    df = pd.DataFrame(task_data)

    # Optimize DataFrame operations
    days_of_week = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df_pivot = df.pivot_table(index='Task ID', columns='Day', values='Duration', aggfunc='sum', fill_value=0)
    df_pivot = df_pivot.reindex(columns=days_of_week, fill_value=0)
    df_pivot = df_pivot.div(3600000).round(2)  # Convert milliseconds to hours

    df = df.drop(['Duration', 'Date', 'Day'], axis=1).drop_duplicates(subset='Task ID')
    df = df.merge(df_pivot, on='Task ID', how='left')
    
    ### OTHER Checks ###
    # define the API parameters
    headers = {"Authorization": __auth__}

    # iterate over the unique task IDs in the dataframe
    for task_id in df['Task ID'].unique():
        # construct the API URL for the task ID
        url = f"https://api.clickup.com/api/v2/task/{task_id}"

        # make the API request and parse the JSON response
        response = requests.get(url, headers=headers)
        tasks = response.json()

        hrs_mins = convert_milliseconds_to_hours_minutes(tasks.get('time_spent', 0))
        df.loc[df['Task ID'] == task_id,
                 'Total (till date)'] = f"{hrs_mins[0]}h {hrs_mins[1]}m"
        # If there is no Custom field just continue
        try:
            for custom_field in tasks.get("custom_fields", []):
                # Process custom field logic
                if 'value' in custom_field and custom_field['type'] == 'drop_down':
                    df.loc[df['Task ID'] == task_id, custom_field['name']] = custom_field['type_config']['options'][custom_field['value']]['name']
        except Exception as e:
            error_message = f"Error processing custom fields for task {task_id}: {e}"
            st.error(error_message)
            # Dump the response JSON nicely for debugging
            st.write("Task response:", json.dumps(tasks, indent=2))


    # Check if 'Proj-Common-Activity' column exists in the DataFrame
    if 'Proj-Common-Activity' in df.columns:
        # Filter out rows where 'Proj-Common-Activity' is 'Vyoma Holiday' or 'Personal Leave'
        df = df[(df['Proj-Common-Activity'] != 'Vyoma Holiday') & (df['Proj-Common-Activity'] != 'Personal Leave')]

    # Check if 'Goal Type' column exists
    if 'Goal Type' not in df.columns:
        # Add a new column with 'nan' values
        df['Goal Type'] = np.nan

    # Initialize a list to collect the names of rows that do not fit the criterion
    rows_with_missing_data = []
    row_id_with_missing_data = []
    rows_missing_goal_type = []
    row_id_missing_goal_type = []
    # Add the new condition to check 'Product' and '-Proj' columns
    tasks_with_missing_proj = []
    task_ids_with_missing_proj = []
    project_columns_check = list(set(df.columns.tolist()).intersection(columns_to_check))
    project_columns = [item for item in project_columns_check if 'Proj' in item or 'Activity' in item or 'HR & Admin' in item]
    # Iterate through rows in the DataFrame
    for index, row in df.iterrows():
        # Extract the 'Task Name' column value for the current row
        task_name = row['Task Name']
        task_id = row['Task ID']
        
        # Updated code to handle both 'nan' strings and np.nan values
        if all(is_nan(row[col]) for col in project_columns_check):  # All specified columns are NaN
            rows_with_missing_data.append(task_name)
            row_id_with_missing_data.append(task_id)
        
        if is_nan(row['Goal Type']):  # Goal Type is NaN
            rows_missing_goal_type.append(task_name)
            row_id_missing_goal_type.append(task_id)
            
        # Check if 'Product' column has a value
        if 'Project ID' in df.columns and not is_nan(row['Project ID']):            
            # Check if any column ending with '-Proj' has a value
            project_set = False
            for col in project_columns:
                if (col in project_columns) and not is_nan(row[col]):
                    project_set = True
                    break            
            # If no '-Proj' column has a value, collect the task details
            if not project_set:
                tasks_with_missing_proj.append(row['Task Name'])
                task_ids_with_missing_proj.append(row['Task ID']) 

    # Output the names of rows that do not fit the criterion
    if rows_with_missing_data or rows_missing_goal_type or tasks_with_missing_proj:
        st.error("Some tasks are missing required information.")
        if rows_with_missing_data:
            st.write("‘Project/Product/Course/Website’ is not set for the below task(s):")
            # st.write(columns_to_check)
            for link_text, link_url in zip(rows_with_missing_data, row_id_with_missing_data):
                st.write(f"[{link_text}](https://app.clickup.com/t/{link_url})")
        if rows_missing_goal_type:
            st.write("Goal Type not set for:")
            for link_text, link_url in zip(rows_missing_goal_type, row_id_missing_goal_type):
                st.write(f"[{link_text}](https://app.clickup.com/t/{link_url})")        
        if tasks_with_missing_proj:
            st.error("Some tasks have 'Product' selected but no project set.")
            st.write("Please update the project information for the following tasks:")
            for task_name, task_id in zip(tasks_with_missing_proj, task_ids_with_missing_proj):
                st.write(f"[{task_name}](https://app.clickup.com/t/{task_id})")
            st.info("You can try generating your timesheet again once you set the above information in these tasks.")
        return
    
    # Add total tracked time for the week
    df['Total (this week)'] = df[days_of_week].sum(axis=1)

    # Add totals row
    totals = df[days_of_week].sum(axis=0)
    df = pd.concat([df, totals.to_frame().T], ignore_index=True)
    weekly_total = df.iloc[-1].sum()
    st.session_state['weekly_total'] = weekly_total
    df.at[df.index[-1], 'Task Status'] = 'Daily Totals ->'
    empty_row = pd.Series([np.nan] * len(df.columns), index=df.columns)
    df = pd.concat([df, empty_row.to_frame().T], ignore_index=True)
    df.iloc[-1, 5] = weekly_total

    days_diff = (end_date - start_date).days + 1
    if days_diff <= 7:
        df.iloc[:, 3] = df.iloc[:, 3].astype(object)
        df.iloc[-1, 3] = "Week's total ="
        week_number = end_date.isocalendar()[1]
        st.session_state['week_number'] = week_number
        df.at[df.index[-1], 'Task Name'] = f'Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}'
    else:
        df.iloc[-1, 3] = 'Total Hours'
        df.at[df.index[-1], 'Task Name'] = f'{start_date_str}, {year_str} - {end_date_str}, {year_str}'

    # Reorder column: move 'Total (this week)'  to the 11th position.
    df.insert(10, 'Total (this week)', df.pop('Total (this week)'))

    # Write to Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        worksheet = writer.sheets['Sheet1']
        for row_num, value in enumerate(df['Task ID'], start=1):
            if pd.isna(value):
                break
            worksheet.write_url(row_num, df.columns.get_loc('Task ID'), f'https://app.clickup.com/t/{value}', string=value)

    processed_data = output.getvalue()
    st.success(f"Successfully generated the timesheet: {filename}")
    st.download_button(
        label="Download Excel File",
        data=processed_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write(f"Total Hours for this time frame: {weekly_total:.2f}")
    st.write(f"Processing Time: {time.time() - start_time_process:.2f} seconds")    
    
    if open_google_sheet:
        st.write("[Open Google Sheet for TS Submission Status](https://docs.google.com/spreadsheets/d/1XLDSTT5m952eiOXhiUtxldIIoEAfQgiVKv5XY2HFOBg/edit?usp=sharing)")    
    
    # 1) replace <NA> with blanks
    df = df.fillna('')    
    df = df.replace(['nan', 'na'], '', regex=True)
    return df

# Function to calculate the default start and end dates based on today's date
def calculate_default_dates():
    today = datetime.today()

    # Calculate the previous Friday
    days_to_friday = (today.weekday() - 4) % 7  # Friday is the 4th day in Python's weekday system
    end_date = today - timedelta(days=days_to_friday)
    
    # If today is Friday, set end date to today
    if today.weekday() == 4:
        end_date = today

    # Calculate the Saturday before the selected Friday
    start_date = end_date - timedelta(days=6)  # Saturday is one day before Friday

    return start_date, end_date

def open_url_in_new_tab(url):
    js_code = f"""
    <script type="text/javascript">
        window.open("{url}", "_blank");
    </script>
    """
    html(js_code)
    
# Function to send email with the table formatted as plain text
def send_email_with_table(df, to_email, cc_email, week_number, start_date_str, 
                          end_date_str, year_str,                          
                          body="Ram ram ram,\nPlease find my weekly timesheet in this Google tracker -"):
        
    subject=f"Timesheet for Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}", 
    # Prepare the subject and body    
    subject = urllib.parse.quote((str(subject[0])).encode('utf-8'))
    body = urllib.parse.quote(body + "\n\n")  # Adding the plain-text table to the body

    # Construct the Gmail URL with the recipient, subject, body, and CC fields pre-filled
    gmail_url = f"https://mail.google.com/mail/?view=cm&to={to_email}&cc={cc_email}&su={subject}&body={body}"

    open_url_in_new_tab(gmail_url)
    
# Streamlit UI
def main():
    st.set_page_config(page_title="Timesheet Generator", page_icon=":calendar:", layout="wide")
    
    # Create two columns: one for the logo and one for the title
    col1, col2 = st.columns([1, 2])  # Adjust the column ratios as needed

    # Display Image in the first column
    with col1:
        image_url = "https://digitalsanskritguru.com/wp-content/uploads/2020/05/Vyoma_Logo_Blue_500x243.png"
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        image = image.resize((167, 81))
        st.image(image, use_container_width=False)

    # Display Title in the second column (centered)
    with col2:
        st.markdown("<h1 style='text-align: left;'>Timesheet Generator</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if "timesheet" not in st.session_state:
        st.session_state["timesheet"] = None
        st.session_state['start_date_str'] = None
        st.session_state['end_date_str'] = None
        st.session_state['year_str'] = None
        st.session_state['week_number'] = None
    if "submit_clicked" not in st.session_state:
        st.session_state["submit_clicked"] = False
    if "download_clicked" not in st.session_state:
        st.session_state["download_clicked"] = False

    # Employee Key Entry
    key = st.text_input("Employee ID: (e.g., C047)")
    
    # Get default start and end dates based on today's date
    default_start_date, default_end_date = calculate_default_dates()

    # Date Selection with Display in 'Day, dd Mon yyyy' Format after Selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start_date)
        # Format the selected start date
        formatted_start_date = start_date.strftime('%a, %d %b %Y')
        # Display the formatted start date just below the selection
        st.write(f"Selected Start Date: {formatted_start_date}")
        to_email = st.text_input("To (Recipient Email Address)")

    with col2:
        end_date = st.date_input("End Date", default_end_date)
        # Format the selected end date
        formatted_end_date = end_date.strftime('%a, %d %b %Y')
        # Display the formatted end date just below the selection
        st.write(f"Selected End Date: {formatted_end_date}")
        cc_email = st.text_input("CC (CC Email Address)", 
                                 "srilatha.vyoma@gmail.com, hr@vyomalabs.in")  # Prefill CC field    

    # Create checkboxes
    open_google_sheet = st.checkbox("Open the Google sheet for TS Submission Status")    

    # Button to generate the timesheet
    if st.button("Generate Timesheet"):
        # Generate the DataFrame and store it in session state
        st.session_state["timesheet"] = get_selected_dates(start_date, end_date, key, open_google_sheet, to_email, cc_email)        
        
    # Check if the timesheet exists in session state
    if st.session_state["timesheet"] is not None:
        # Display the timesheet as a table   
        # cell‐wise formatter
        def fmt_cell(x):
            if isinstance(x, (int, float, np.floating, np.integer)):
                return f"{x:.2f}"
            return x
        # 3) build the Styler
        styled = (
            st.session_state["timesheet"].fillna('')  # blank out the <NA>s
              .style
              .format(fmt_cell)                # apply fmt_cell to each cell
               .set_table_attributes('style="width:100%; table-layout: auto; font-size: .75vw"')
               .to_html()
        )
        st.markdown(styled, unsafe_allow_html=True)        
        
        st.session_state["submit_clicked"] = False
        st.session_state["download_clicked"] = False
        st.success("Timesheet generated successfully!")
        # Submit button
        if st.button("Send E-mail"):            
            send_email_with_table(st.session_state["timesheet"], to_email, cc_email,
                                   st.session_state['week_number'],
                                   st.session_state['start_date_str'],
                                   st.session_state['end_date_str'],
                                   st.session_state['year_str'])
            st.session_state["submit_clicked"] = True
            st.session_state["download_clicked"] = False                                        
    else:
        st.info("Click 'Generate Timesheet' to create a timesheet.")

    # Footer
    st.write(f"Version {__version__} {__date__}")

if __name__ == "__main__":
    main()
