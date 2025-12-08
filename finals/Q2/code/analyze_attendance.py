import pandas as pd
import datetime

def parse_date(date_str):
    # Format appears to be YY/DD/MM from the inspection
    # e.g., 25/19/08 -> 2025-08-19
    # The file has: 25/19/08, 25/21/08...
    # Wait, let's look at the data again.
    # 25/19/08 -> Aug 19, 2025? (Year/Day/Month)
    # The instructions imply Fall 2025.
    # Let's verify with known dates.
    # Quiz 2: Oct 7.
    # Let's look for a date that could be Oct 7.
    # If format is YY/DD/MM, then 25/07/10 is Oct 7, 2025.
    # If format is YY/MM/DD, then 25/19/08 is invalid month 19.
    # So YY/DD/MM seems correct.
    try:
        parts = date_str.split('/')
        year = int(parts[0]) + 2000
        day = int(parts[1])
        month = int(parts[2])
        return datetime.date(year, month, day)
    except Exception as e:
        print(f"Error parsing date: {date_str} - {e}")
        return None

def main():
    # Assuming script is run from c:\Users\tysup\Documents\CSCE_580
    file_path = "finals/data_uniform.csv"
    output_path = "finals/Q2/results.txt"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates
    df['ParsedDate'] = df['Date'].apply(parse_date)
    
    with open(output_path, 'w') as f:
        # Q2c.a: Number of classes and their dates
        unique_dates = sorted(df['ParsedDate'].unique())
        num_classes = len(unique_dates)
        f.write(f"Q2c.a: Number of classes: {num_classes}\n")
        f.write("Dates:\n")
        for d in unique_dates:
            f.write(f"  {d.strftime('%Y-%m-%d')}\n")
        f.write("-" * 20 + "\n")

        # Q2c.b: Median class attendance per class
        # Count rows per date
        attendance_per_class = df.groupby('ParsedDate').size()
        median_attendance = attendance_per_class.median()
        f.write(f"Q2c.b: Median class attendance: {median_attendance}\n")
        # f.write("Attendance per class:\n")
        # f.write(str(attendance_per_class) + "\n")
        f.write("-" * 20 + "\n")
        
        # Q2c.c: Dates with lowest and highest attendance
        min_attendance = attendance_per_class.min()
        max_attendance = attendance_per_class.max()
        
        lowest_dates = attendance_per_class[attendance_per_class == min_attendance].index.tolist()
        highest_dates = attendance_per_class[attendance_per_class == max_attendance].index.tolist()
        
        f.write(f"Q2c.c: Lowest attendance ({min_attendance}): {[d.strftime('%Y-%m-%d') for d in lowest_dates]}\n")
        f.write(f"Highest attendance ({max_attendance}): {[d.strftime('%Y-%m-%d') for d in highest_dates]}\n")
        f.write("-" * 20 + "\n")

        # Q2c.d: Correlation with evaluation dates
        eval_dates = [
            datetime.date(2025, 10, 7),
            datetime.date(2025, 11, 11),
            datetime.date(2025, 11, 18)
        ]
        
        f.write("Q2c.d: Correlation check\n")
        for d in eval_dates:
            att = attendance_per_class.get(d, "No Class")
            f.write(f"  Eval Date {d}: Attendance = {att}\n")
            
        avg_attendance = attendance_per_class.mean()
        f.write(f"  Average Attendance: {avg_attendance:.2f}\n")
    
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()
