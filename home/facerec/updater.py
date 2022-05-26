from datetime import datetime
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from .train_faces import trainer
import time

def start():
    pass
    # schedule.every().day.at("11:18").do(trainer)
    # print("Scheduled Re-Training is starting now...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(5)
    # if True:
    #     pass
    # else:
    #     pass
    # schedule=BackgroundScheduler()
    # schedule.add_job(trainer,'interval',seconds=1)
    # schedule.start()
