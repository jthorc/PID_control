import tkinter as tk
from tkinter import ttk
import time
import threading

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.05, integral_limit_pos=100.0, integral_limit_neg=-100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()
        self.integral_limit_pos = integral_limit_pos
        self.integral_limit_neg = integral_limit_neg

    def update(self, setpoint, measurement):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6

        error = setpoint - measurement
        self.integral += error * dt
        # Anti-windup: clamp integral
        self.integral = max(min(self.integral, self.integral_limit_pos), self.integral_limit_neg)

        derivative = (error - self.prev_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.prev_error = error
        self.last_time = now
        return output


class SimpleThermalSystem:
    def __init__(self, initial_temp=20.0, ambient_temp=20.0, cooling_rate=0.02):
        self.temperature = initial_temp
        self.ambient = ambient_temp
        self.cooling_rate = cooling_rate

    def step(self, power_watts, dt=0.1):
        self.temperature += power_watts * 0.01 * dt
        self.temperature += (self.ambient - self.temperature) * self.cooling_rate * dt
        return self.temperature


class PIDTemperatureApp:
    def __init__(self, master):
        self.master = master
        master.title("PID Temperature Control")

        control_frame = ttk.Frame(master)
        control_frame.pack(padx=8, pady=8, fill="x")

        ttk.Label(control_frame, text="Kp:").grid(row=0, column=0, sticky="w")
        ttk.Label(control_frame, text="Ki:").grid(row=1, column=0, sticky="w")
        ttk.Label(control_frame, text="Kd:").grid(row=2, column=0, sticky="w")

        self.kp_var = tk.DoubleVar(value=1.0)
        self.ki_var = tk.DoubleVar(value=0.1)
        self.kd_var = tk.DoubleVar(value=0.05)

        tk.Scale(control_frame, from_=0, to=10, resolution=0.01, orient="horizontal", variable=self.kp_var, length=200).grid(row=0, column=1)
        tk.Scale(control_frame, from_=0, to=5, resolution=0.01, orient="horizontal", variable=self.ki_var, length=200).grid(row=1, column=1)
        tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.kd_var, length=200).grid(row=2, column=1)

        ttk.Label(control_frame, text="Integral Max:").grid(row=0, column=2, padx=10)
        self.integral_max_var = tk.DoubleVar(value=100.0)
        tk.Scale(control_frame, from_=0, to=500, resolution=1, orient="horizontal", variable=self.integral_max_var, length=200).grid(row=0, column=3)

        ttk.Label(control_frame, text="Integral Min:").grid(row=1, column=2, padx=10)
        self.integral_min_var = tk.DoubleVar(value=-100.0)
        tk.Scale(control_frame, from_=-500, to=0, resolution=1, orient="horizontal", variable=self.integral_min_var, length=200).grid(row=1, column=3)

        ttk.Label(control_frame, text="Target Temp (C):").grid(row=2, column=2, padx=10)
        self.temperature_target_var = tk.DoubleVar(value=50.0)
        tk.Scale(control_frame, from_=0, to=200, resolution=1, orient="horizontal", variable=self.temperature_target_var, length=200).grid(row=2, column=3)

        ttk.Label(control_frame, text="Power Limit (W):").grid(row=3, column=0, padx=10)
        self.power_limit_var = tk.DoubleVar(value=200.0)
        tk.Scale(control_frame, from_=10, to=500, resolution=5, orient="horizontal", variable=self.power_limit_var, length=200).grid(row=3, column=1)

        ttk.Label(control_frame, text="Initial Temp (C):").grid(row=3, column=2, padx=10)
        self.initial_temp_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=100, resolution=1, orient="horizontal", variable=self.initial_temp_var, length=200).grid(row=3, column=3)

        ttk.Label(control_frame, text="Natural Cooling Rate:").grid(row=4, column=0, sticky="w", pady=5)
        self.cooling_rate_var = tk.DoubleVar(value=0.02)
        tk.Scale(control_frame, from_=0, to=0.1, resolution=0.001, orient="horizontal", variable=self.cooling_rate_var, length=200).grid(row=4, column=1, sticky="w")

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start)
        self.start_button.grid(row=4, column=2, padx=6, pady=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=4, column=3, pady=5)

        self.current_temp_var = tk.StringVar(value="Temp: 0.0 C")
        self.current_power_var = tk.StringVar(value="Power: 0.0 W")
        ttk.Label(control_frame, textvariable=self.current_temp_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(control_frame, textvariable=self.current_power_var).grid(row=5, column=2, columnspan=2, sticky="w", pady=5)

        fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_temp = fig.add_subplot(211)
        self.ax_temp.set_title("Temperature vs Time")
        fig.subplots_adjust(hspace=0.4)
        self.ax_power = fig.add_subplot(212)
        self.ax_power.set_title("Power Output vs Time")

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.time_data = []
        self.temp_data = []
        self.power_data = []
        self.heat_power_data = []
        self.cool_power_data = []

        self.system = SimpleThermalSystem(initial_temp=self.initial_temp_var.get(), cooling_rate=self.cooling_rate_var.get())
        self.pid = PID(kd=self.kd_var.get(), integral_limit_pos=self.integral_max_var.get(), integral_limit_neg=self.integral_min_var.get())

        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.system = SimpleThermalSystem(initial_temp=self.initial_temp_var.get(), cooling_rate=self.cooling_rate_var.get())
            self.time_data.clear()
            self.temp_data.clear()
            self.power_data.clear()
            self.heat_power_data.clear()
            self.cool_power_data.clear()
            self.thread = threading.Thread(target=self.control_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def control_loop(self):
        t0 = time.time()
        while self.running:
            self.pid.kp = self.kp_var.get()
            self.pid.ki = self.ki_var.get()
            self.pid.kd = self.kd_var.get()
            self.pid.integral_limit_pos = self.integral_max_var.get()
            self.pid.integral_limit_neg = self.integral_min_var.get()

            setpoint = self.temperature_target_var.get()
            max_power = abs(self.power_limit_var.get())

            current_temp = self.system.temperature
            self.system.cooling_rate = self.cooling_rate_var.get()

            power = self.pid.update(setpoint, current_temp)
            power = max(min(power, max_power), -max_power)

            temp = self.system.step(power)

            self.current_temp_var.set(f"Temp: {temp:.2f} C")
            self.current_power_var.set(f"Power: {power:.2f} W")

            elapsed = time.time() - t0
            self.time_data.append(elapsed)
            self.temp_data.append(temp)
            self.power_data.append(power)

            self.heat_power_data.append(power if power > 0 else 0)
            self.cool_power_data.append(-power if power < 0 else 0)

            self.update_plots()
            time.sleep(0.1)

    def update_plots(self):
        self.ax_temp.clear()
        self.ax_power.clear()
        self.ax_temp.plot(self.time_data, self.temp_data, label="Current Temp")
        self.ax_temp.plot(self.time_data, [self.temperature_target_var.get()]*len(self.time_data), 'r--', label="Target Temp")
        self.ax_temp.plot(self.time_data, [self.system.ambient]*len(self.time_data), 'g--', label="Ambient Temp")
        self.ax_temp.set_ylabel("Temp (C)")
        self.ax_temp.legend()
        self.ax_temp.grid(True)

        self.ax_power.plot(self.time_data, self.heat_power_data, 'r', label="Heating Power")
        self.ax_power.plot(self.time_data, self.cool_power_data, 'b', label="Cooling Power")
        self.ax_power.set_ylabel("Power (W)")
        self.ax_power.set_xlabel("Time (s)")
        self.ax_power.legend()
        self.ax_power.grid(True)

        try:
            self.canvas.draw()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = PIDTemperatureApp(root)
    root.mainloop()