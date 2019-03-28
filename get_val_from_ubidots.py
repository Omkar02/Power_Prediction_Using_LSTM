from ubidots import ApiClient
import time



api = ApiClient("BBFF-31019c8d2021bf324cbaf0d8db74df7d33f")

current = api.get_variable("5c97c3da93f3c376315b026c")
voltage = api.get_variable("5c97c3da93f3c376315b026b")
graph = api.get_variable("5c9847d31d8472615c845a0f")


def val_sender_ubi(the_val,result_m):
    stuff_1 = the_val + result_m
    graph.save_value({'value': stuff_1})


def get_val_ubi(result_mean):
    while True:
        temp = []
        while True:
            cur_time = current.get_values(5)[0]['timestamp']
            print('C', (time.strftime("%a %d %b %Y %H:%M:%S Local_Time", time.localtime(cur_time / 1000.0))))
            # vol_time = voltage.get_values(5)[0]['timestamp']

            cur_value = current.get_values(5)[0]['value']
            vol_value = voltage.get_values(5)[0]['value']
            power = cur_value * vol_value
            temp.append(power - result_mean)
            if len(temp) == 10:
                break
        print(temp)
        yield temp
