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

api_key = st.secrets["auth"]
team_id = st.secrets["team_id"]

__version__ = "v3.0.1"
__date__ = "22nd November 2024"
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
    "Admin-Project", "Linguistic-Activity"
]
project_columns = [
    "Proj-Common-Activity", "Proj-Outside-Office",
    "Management-Project", "Technology-Project", "Linguistic-Project",
    "MMedia-Project", "Project-CST", "Sales-Mktg-Project", "Project-ELA",
    "Proj-KidsPersona", "Project-Finance", "SFH-Admin-Project",
    "Admin-Project", "Linguistic-Activity"
]

# Create a timezone object for IST
ist_timezone = pytz.timezone('Asia/Kolkata')

# Exchange keys and values
month_flipped = {value: key for key, value in month_dict.items()}

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

def open_link(link):
    st.write(f"[Open Task in ClickUp](https://app.clickup.com/t/{link})")

def get_selected_dates(start_date, end_date, key, open_google_sheet):
    if not key:
        st.error("Please enter your Employee ID.")
        return

    key = key.upper()

    # Format dates
    start_date_str = start_date.strftime("%b %d")
    end_date_str = end_date.strftime("%b %d")
    year_str = str(start_date.year)

    # Generate filename
    filename = f"{key}_{start_date_str}_to_{end_date_str}_{year_str}.xlsx"

    # Retrieve information from ClickUp
    start_time_process = time.time()

    members_dict = memberInfo()
    if key not in members_dict:
        st.error("Invalid Employee ID. Please check and try again.")
        return

    employee_key = members_dict[key]  # Convert our key to ClickUp key

    # start_timestamp = int(start_date.replace(tzinfo=timezone.utc).timestamp())
    # end_timestamp = int(end_date.replace(tzinfo=timezone.utc).timestamp())
    # Convert start_date to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    start_timestamp = int(start_datetime.replace(tzinfo=timezone.utc).timestamp())
    
    # Convert end_date to datetime
    end_datetime = datetime.combine(end_date, datetime.min.time())
    end_timestamp = int(end_datetime.replace(tzinfo=timezone.utc).timestamp())

    
    url = f"https://api.clickup.com/api/v2/team/{team_id}/time_entries"
    query = {
        "start_date": str(int(start_timestamp - 19800) * 1000),  # Converting to milliseconds from seconds
        "end_date": str(int((end_timestamp + 86399) * 1000) - 19800000),
        "assignee": employee_key,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": __auth__
    }

    response = requests.get(url, headers=headers, params=query)
    data = response.json()

    if 'data' not in data or not data['data']:
        st.error("There are no entries in this Date Range.\nPlease change Date Range or Update Entries in ClickUp.")
        return

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
        start_time = int(entry['start']) // 1000  # Convert to seconds

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

    # define the API parameters
    headers = {"Authorization": __auth__}

    # iterate over the unique task IDs in the dataframe
    for task_id in df_h['Task ID'].unique():
        # construct the API URL for the task ID
        url = f"https://api.clickup.com/api/v2/task/{task_id}"

        # make the API request and parse the JSON response
        response = requests.get(url, headers=headers)
        tasks = response.json()

        hrs_mins = convert_milliseconds_to_hours_minutes(tasks.get('time_spent', 0))
        df_h.loc[df_h['Task ID'] == task_id,
                 'Total Time tracked for this task till now (hrs)'] = f"{hrs_mins[0]}h {hrs_mins[1]}m"
        # If there is no Custom field just continue
        try:
            # iterate over the custom fields for the task
            # for custom_field in tasks.get('custom_fields', []):
            #     if 'value' in custom_field:
            #         if custom_field['type'] == 'drop_down':
            #             # set the value in the dataframe for the current task ID and custom field name
            #             option_id = custom_field['value']
            #             options = custom_field['type_config']['options']
            #             option_name = next((opt['name'] for opt in options if opt['id'] == option_id), None)
            #             df_h.loc[df_h['Task ID'] == task_id, custom_field['name']] = option_name
            # iterate over the custom fields for the task
            for custom_field in tasks['custom_fields']:
                if 'value' in custom_field:
                    if custom_field['type'] == 'drop_down':
                        # set the value in the dataframe for the current task ID and custom field name
                        df_h.loc[df_h['Task ID'] == task_id, custom_field['name']] = custom_field['type_config']['options'][custom_field['value']]['name']
        except Exception as e:
            st.write(f"Error processing custom fields for task {task_id}: {e}")
            pass

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
    # Add the new condition to check 'Product' and '-Proj' columns
    tasks_with_missing_proj = []
    task_ids_with_missing_proj = []
    columns_to_check = list(set(df_h.columns.tolist()).intersection(columns_to_check))
    # Iterate through rows in the DataFrame
    for index, row in df_h.iterrows():
        # Extract the 'Task Name' column value for the current row
        task_name = row['Task Name']
        task_id = row['Task ID']
        
        # Updated code to handle both 'nan' strings and np.nan values
        if all(is_nan(row[col]) for col in columns_to_check):  # All specified columns are NaN
            rows_with_missing_data.append(task_name)
            row_id_with_missing_data.append(task_id)
        
        if is_nan(row['Goal Type']):  # Goal Type is NaN
            rows_missing_goal_type.append(task_name)
            row_id_missing_goal_type.append(task_id)
            
        # Check if 'Product' column has a value
        if 'Project ID' in df_h.columns and not is_nan(row['Project ID']):            
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
            st.write(columns_to_check)
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

    # Add the 'time_this_week' column by summing the values of all days_of_week columns
    df_h['Total Tracked this week in this task'] = df_h[days_of_week].sum(axis=1)
    # Calculate the totals of the days_of_week columns
    totals = df_h[days_of_week].sum(axis=0)

    # Append totals as a new row to the DataFrame
    df_h = pd.concat([df_h, totals.to_frame().T], ignore_index=True)
    # Sum the values in the last row of the DataFrame
    weekly_total = df_h.iloc[-1].sum()

    # Update the value in the 'Status' column for the last row
    df_h.at[df_h.index[-1], 'Task Status'] = 'Daily Totals ->'
    
    # Create an empty row with NaN values
    empty_row = pd.Series([np.nan] * len(df_h.columns), index=df_h.columns)
    # Append the empty row to the DataFrame
    df_h = pd.concat([df_h, empty_row.to_frame().T], ignore_index=True)    
    # Append a value to the 6th column
    df_h.iloc[-1, 5] = weekly_total

    # Determine if the date range is a week
    days_diff = (end_date - start_date).days + 1
    if days_diff <= 7:
        df_h.iloc[:, 3] = df_h.iloc[:, 3].astype(object)  # Convert the entire column to object type
        df_h.iloc[-1, 3] = 'Week\'s total ='
        week_number = end_date.isocalendar()[1]
        df_h.at[df_h.index[-1], 'Task Name'] = f'Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}'
    else:
        df_h.iloc[-1, 3] = 'Total Hours Tracked ='
        df_h.at[df_h.index[-1], 'Task Name'] = f'{start_date_str}, {year_str} - {end_date_str}, {year_str}'

    # Move the column to the 11th position
    df_h.insert(10, 'Total Tracked this week in this task',
                df_h.pop('Total Tracked this week in this task'))

    # Write the DataFrame to an Excel file in memory
    from io import BytesIO
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_h.to_excel(writer, sheet_name='Sheet1', index=False)

    # Get the xlsxwriter workbook and worksheet objects
    worksheet = writer.sheets['Sheet1']

    # Add hyperlinks to the 'Task ID' column
    for row_num, value in enumerate(df_h['Task ID'], start=1):
        if pd.isna(value):
            break
        url = f'https://app.clickup.com/t/{value}'
        worksheet.write_url(row_num, df_h.columns.get_loc('Task ID'), url, string=value)

    writer.close()
    processed_data = output.getvalue()

    st.success(f"Successfully generated the timesheet: {filename}")
    st.download_button(
        label="Download Excel File",
        data=processed_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    total_time = df_h['Total Tracked this week in this task'].sum()
    st.write(f"Total Hours for this time frame: {total_time}")
    st.write(f"Processing Time: {time.time() - start_time_process} seconds")

    if open_google_sheet:
        url1 = 'https://docs.google.com/spreadsheets/d/1XLDSTT5m952eiOXhiUtxldIIoEAfQgiVKv5XY2HFOBg/edit?usp=sharing'        
        st.write(f"[Open Google Sheet for TS Submission Status]({url1})")

def main():
    st.set_page_config(page_title="Timesheet Generator", page_icon=":calendar:", layout="centered")    

    # Display Image
    image_url = "https://digitalsanskritguru.com/wp-content/uploads/2020/05/Vyoma_Logo_Blue_500x243.png"
    response = requests.get(image_url)
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    image = image.resize((167, 81))
    st.image(image, use_column_width=False)

    st.title("Timesheet Generator")

    # Employee Key Entry
    key = st.text_input("Employee ID: (e.g., C047)")

    # Date Selection with Display in 'Day, dd Mon yyyy' Format after Selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7))
        # Format the selected start date
        formatted_start_date = start_date.strftime('%a, %d %b %Y')
        # Display the formatted start date just below the selection
        st.write(f"Selected Start Date: {formatted_start_date}")

    with col2:
        end_date = st.date_input("End Date", datetime.today())
        # Format the selected end date
        formatted_end_date = end_date.strftime('%a, %d %b %Y')
        # Display the formatted end date just below the selection
        st.write(f"Selected End Date: {formatted_end_date}")

    # Create checkboxes
    open_google_sheet = st.checkbox("Open the Google sheet for TS Submission Status")

    # Submit Button
    if st.button("Submit"):
        get_selected_dates(start_date, end_date, key, open_google_sheet)

    st.write("Note: Please find the generated Excel output by clicking the download button above.")

    # Footer
    st.write(f"Version {__version__} {__date__}")

if __name__ == "__main__":
    main()
