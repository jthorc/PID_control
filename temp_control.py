import tkinter as tk
from tkinter import ttk
import time
import threading

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ======================
# PID Controller (with user-requested clamp-integral condition)
# ======================
class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.05,
                 integral_limit_pos=100.0, integral_limit_neg=-100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()
        self.integral_limit_pos = integral_limit_pos
        self.integral_limit_neg = integral_limit_neg

    def update(self, setpoint, measurement, max_power):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6

        error = setpoint - measurement
        derivative = (error - self.prev_error) / dt

        output_unsat_now = self.kp * error + self.ki * self.integral + self.kd * derivative

        clamp_threshold_pos = 1.0
        clamp_threshold_neg = -1.0

        if (output_unsat_now >= clamp_threshold_pos and error > 0) or \
           (output_unsat_now <= clamp_threshold_neg and error < 0):
            self.integral = max(min(self.integral, self.integral_limit_pos), self.integral_limit_neg)
        else:
            self.integral += error * dt
            self.integral = max(min(self.integral, self.integral_limit_pos), self.integral_limit_neg)

        output_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative

        if measurement >= setpoint and output_unsat > 0:
            output = 0.0
        elif measurement <= setpoint and output_unsat < 0:
            output = 0.0
        else:
            if max_power is not None:
                output = max(min(output_unsat, max_power), -max_power)
            else:
                output = output_unsat

        self.prev_error = error
        self.last_time = now
        return output


# ======================
# Simple Thermal Simulation
# ======================
class SimpleThermalSystem:
    def __init__(self, initial_temp=20.0, ambient_temp=20.0, cooling_rate=0.02):
        self.temperature = initial_temp
        self.ambient = ambient_temp
        self.cooling_rate = cooling_rate

    def step(self, power_watts, dt=0.1):
        self.temperature += power_watts * 0.01 * dt
        self.temperature += (self.ambient - self.temperature) * self.cooling_rate * dt
        return self.temperature


# ======================
# Tkinter App
# ======================
class PIDTemperatureApp:
    def __init__(self, master):
        self.master = master
        master.title("PID Temperature Control")

        # --- Controls Frame ---
        control_frame = ttk.Frame(master)
        control_frame.grid(row=0, column=0, padx=8, pady=8, sticky="ew")

        ttk.Label(control_frame, text="Kp:").grid(row=0, column=0, sticky="w")
        ttk.Label(control_frame, text="Ki:").grid(row=1, column=0, sticky="w")
        ttk.Label(control_frame, text="Kd:").grid(row=2, column=0, sticky="w")

        self.kp_var = tk.DoubleVar(value=1.0)
        self.ki_var = tk.DoubleVar(value=0.1)
        self.kd_var = tk.DoubleVar(value=0.05)

        tk.Scale(control_frame, from_=0, to=10, resolution=0.01, orient="horizontal",
                 variable=self.kp_var, length=200).grid(row=0, column=1)
        tk.Scale(control_frame, from_=0, to=5, resolution=0.01, orient="horizontal",
                 variable=self.ki_var, length=200).grid(row=1, column=1)
        tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient="horizontal",
                 variable=self.kd_var, length=200).grid(row=2, column=1)

        ttk.Label(control_frame, text="Integral Max:").grid(row=0, column=2, padx=10)
        self.integral_max_var = tk.DoubleVar(value=100.0)
        tk.Scale(control_frame, from_=0, to=500, resolution=1, orient="horizontal",
                 variable=self.integral_max_var, length=200).grid(row=0, column=3)

        ttk.Label(control_frame, text="Integral Min:").grid(row=1, column=2, padx=10)
        self.integral_min_var = tk.DoubleVar(value=-100.0)
        tk.Scale(control_frame, from_=-500, to=0, resolution=1, orient="horizontal",
                 variable=self.integral_min_var, length=200).grid(row=1, column=3)

        ttk.Label(control_frame, text="Target Temp (C):").grid(row=2, column=2, padx=10)
        self.temperature_target_var = tk.DoubleVar(value=50.0)
        tk.Scale(control_frame, from_=0, to=200, resolution=1, orient="horizontal",
                 variable=self.temperature_target_var, length=200).grid(row=2, column=3)

        ttk.Label(control_frame, text="Power Limit (W):").grid(row=3, column=0, padx=10)
        self.power_limit_var = tk.DoubleVar(value=200.0)
        tk.Scale(control_frame, from_=10, to=500, resolution=5, orient="horizontal",
                 variable=self.power_limit_var, length=200).grid(row=3, column=1)

        ttk.Label(control_frame, text="Initial Temp (C):").grid(row=3, column=2, padx=10)
        self.initial_temp_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=100, resolution=1, orient="horizontal",
                 variable=self.initial_temp_var, length=200).grid(row=3, column=3)

        ttk.Label(control_frame, text="Cooling Rate:").grid(row=4, column=0, sticky="w", pady=5)
        self.cooling_rate_var = tk.DoubleVar(value=0.02)
        tk.Scale(control_frame, from_=0.001, to=0.1, resolution=0.001, orient="horizontal",
                 variable=self.cooling_rate_var, length=200).grid(row=4, column=1, sticky="w")

        # --- Buttons Frame ---
        buttons_frame = ttk.Frame(master)
        buttons_frame.grid(row=1, column=0, pady=5, sticky="ew")

        self.start_button = ttk.Button(buttons_frame, text="Start/Resume", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="w", padx=5)
        self.pause_button = ttk.Button(buttons_frame, text="Pause", command=self.pause)
        self.pause_button.grid(row=0, column=1, sticky="w", padx=5)
        self.reset_button = ttk.Button(buttons_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=2, sticky="w", padx=5)

        # --- Display Frame ---
        display_frame = ttk.Frame(master)
        display_frame.grid(row=2, column=0, pady=5, sticky="ew")

        self.current_temp_var = tk.StringVar(value="Temp: 0.0 C")
        self.current_power_var = tk.StringVar(value="Power: 0.0 W")
        ttk.Label(display_frame, textvariable=self.current_temp_var).grid(row=0, column=0, padx=5, sticky="w")
        ttk.Label(display_frame, textvariable=self.current_power_var).grid(row=0, column=1, padx=5, sticky="w")

        # --- Plot Frame (FIXED) ---
        plot_frame = ttk.Frame(master)
        plot_frame.grid(row=3, column=0, sticky="nsew")
        master.grid_rowconfigure(3, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # allow canvas expansion
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_temp = fig.add_subplot(211)
        self.ax_temp.set_title("Temperature vs Time")
        self.ax_power = fig.add_subplot(212)
        self.ax_power.set_title("Power vs Time")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35)

        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- Data & System ---
        self.time_data, self.temp_data, self.power_data = [], [], []
        self.heat_power_data, self.cool_power_data = [], []
        self.system = SimpleThermalSystem(initial_temp=self.initial_temp_var.get(),
                                          cooling_rate=self.cooling_rate_var.get())
        self.pid = PID(kd=self.kd_var.get(),
                       integral_limit_pos=self.integral_max_var.get(),
                       integral_limit_neg=self.integral_min_var.get())
        self.running, self.paused = False, False

    def start(self):
        if self.running:
            self.paused = False
        else:
            self.running = True
            self.paused = False
            self.system = SimpleThermalSystem(initial_temp=self.initial_temp_var.get(),
                                              cooling_rate=self.cooling_rate_var.get())
            self.pid.integral = 0.0
            self.pid.prev_error = 0.0
            self.pid.last_time = time.time()
            self.thread = threading.Thread(target=self.control_loop, daemon=True)
            self.thread.start()

    def pause(self):
        self.paused = True

    def reset(self):
        self.running = False
        self.paused = False
        time.sleep(0.1)
        self.system = SimpleThermalSystem(initial_temp=self.initial_temp_var.get(),
                                          cooling_rate=self.cooling_rate_var.get())

        self.pid.integral, self.pid.prev_error = 0.0, 0.0
        self.pid.last_time = time.time()

        self.time_data.clear()
        self.temp_data.clear()
        self.power_data.clear()
        self.heat_power_data.clear()
        self.cool_power_data.clear()

        self.update_plots()
        self.current_temp_var.set(f"Temp: {self.system.temperature:.2f} C")
        self.current_power_var.set("Power: 0.0 W")

    def control_loop(self):
        t0 = time.time()
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            self.pid.kp = self.kp_var.get()
            self.pid.ki = self.ki_var.get()
            self.pid.kd = self.kd_var.get()
            self.pid.integral_limit_pos = self.integral_max_var.get()
            self.pid.integral_limit_neg = self.integral_min_var.get()

            setpoint = self.temperature_target_var.get()
            max_power = abs(self.power_limit_var.get())
            self.system.cooling_rate = self.cooling_rate_var.get()

            power = self.pid.update(setpoint, self.system.temperature, max_power)

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
        self.ax_temp.plot(self.time_data,
                          [self.temperature_target_var.get()] * len(self.time_data),
                          'r--', label="Target Temp")
        self.ax_temp.plot(self.time_data,
                          [self.system.ambient] * len(self.time_data),
                          'g--', label="Ambient Temp")
        self.ax_temp.set_ylabel("Temp (C)")
        self.ax_temp.legend()
        self.ax_temp.grid(True)

        self.ax_power.plot(self.time_data, self.heat_power_data, label="Heating Power")
        self.ax_power.plot(self.time_data, self.cool_power_data, label="Cooling Power")
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
    root.geometry("900x700")
    app = PIDTemperatureApp(root)
    root.mainloop()
