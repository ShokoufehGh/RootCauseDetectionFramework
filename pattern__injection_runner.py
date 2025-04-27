from patterns.pattern1_sytem_user import main as p1
from patterns.pattern2_peak_time import main as p2
from patterns.pattern3_individual_typo import main as p3
from patterns.pattern4_overall_workload import main as p4
from patterns.pattern5_individual_convention import main as p5
from patterns.pattern6_workload_monthly import main as p6
from patterns.pattern7_department import main as p7
import sys
from datetime import datetime

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log.write(message)       # Save to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def pattern1_sytem_user():
    activity_label = 'Record Goods Receipt' 
    percentages = [0.35, 0.2, 0.2, 0.25]    
    synonyms = [
                activity_label,
                'Receive Shipment', 
                'Goods Delivery Confirmation',
                'Goods Received Log'
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_batch.csv"
    p1(percentages, synonyms, csv_url, output_csv)

def pattern2_peak_time_week_day():
    activity_label = 'Clear Invoice'
    percentages = [0.85, 0.05, 0.05, 0.05]
    
    synonyms = [
                activity_label,
                "Clear Invoce",
                "Cleer Invoice",
                "Clear Invoise"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    period="week_day"
    output_csv = f"data/updated_peak_{period}.csv"
    p2(percentages, synonyms, csv_url, output_csv, period)
    
def pattern2_peak_time_month_day():
    activity_label = 'Record Invoice Receipt'
    percentages = [0.85, 0.05, 0.05, 0.05]
    
    synonyms = [
                activity_label,
                "Record Invoice Receit",
                "Record Invoice Reciept",
                "Record Invoise Receipt"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    period="month_day"
    output_csv = f"data/updated_peak_{period}.csv"
    p2(percentages, synonyms, csv_url, output_csv, period)
    
def pattern2_peak_time_year_month():
    activity_label = 'Create Purchase Requisition Item'
    percentages = [0.85, 0.05, 0.05, 0.05]
    
    synonyms = [
                activity_label,
                "Create Purchse Requisition Item",
                "Create Purchase Requisision Item-",
                "Create Purchase Requisision Item"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    period="year_month"
    output_csv = f"data/updated_peak_{period}.csv"
    p2(percentages, synonyms, csv_url, output_csv, period)
    
def pattern3_individual_typo():
    activity_label = 'Cancel Invoice Receipt'
    percentage_user = 0.20
    percentage_update = 0.05
    synonyms = [
                activity_label,
                "Cancel Invoice Receit",
                "Cancel Invoice Reciept",
                "Cancel Invoise Receipt"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_individual_typo.csv"
    
    p3(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv) 
    
def pattern4_overall_workload():
    activity_label = 'Change Quantity'
    # percentages = [0.95, 0.03, 0.01, 0.01]    
    percentage_user = 0.1
    percentage_update = 0.05
    synonyms = [
                activity_label,
                "Change Quntity",
                "Change Qantity",
                "ChangeQuantity"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_overall_workload.csv"
    p4(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv) 
    
def pattern5_individual_convention():
    activity_label = 'Remove Payment Block' 
    percentages = [0.35, 0.2, 0.2, 0.25]    
    synonyms = [
                activity_label,
                'Clear Payment Block', 
                'Eliminate Payment',
                'Cancel Payment Block'
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_individual_convention.csv"
    p5(activity_label, percentages, synonyms, csv_url, output_csv)

def pattern6_workload_monthly():
    
    activity_label = 'Delete Purchase Order Item'
    # percentages = [0.95, 0.03, 0.01, 0.01]    
    percentage_user = 0.20
    percentage_update = 0.08
    minimum_day_activity = 5
    synonyms = [
                activity_label,
                "Delete Purchase Order Itm",
                "Delete Parchase Order Item",
                "Delete Purchas Order Item"
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_workload_monthly.csv"
    csv_url = "data/final_updated_1222345.csv"
    output_csv = "data/final_updated_12223456.csv"
    p6(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv,minimum_day_activity) 

def pattern7_department():
    activity_label = 'Record Goods Receipt' 
    percentages = [0.35, 0.2, 0.2, 0.25]    
    synonyms = [
                activity_label,
                'Receive Shipment', 
                'Goods Delivery Confirmation',
                'Goods Received Log'
            ]
    csv_url = "data/BPI_Challenge_2019.csv"
    output_csv = "data/updated_batch.csv"
    p7(activity_label, percentages, synonyms, csv_url, output_csv) 

def runWithLog(func, log_postfix=""):
    log_file = f"log/{func.__name__}{log_postfix}.txt"
    sys.stdout = Logger(log_file)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
    
    func()
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    
def main():   
    # runWithLog(pattern1_sytem_user)
    # runWithLog(pattern2_peak_time_week_day)
    # runWithLog(pattern2_peak_time_month_day)
    # runWithLog(pattern2_peak_time_year_month)
    # runWithLog(pattern3_individual_typo)
    # runWithLog(pattern4_overall_workload)
    # runWithLog(pattern5_individual_convention)
    # runWithLog(pattern6_workload_monthly, "_final")
    return

main()
