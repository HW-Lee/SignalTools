import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler, MouseButton
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt

from librosa.core import load as audioload
from scipy.fftpack import fft
import gc

import numpy as np
import os

from tictoc import TicToc

GLOBAL = {}
matplotlib.rcParams["agg.path.chunksize"] = 10000

class MainWindow(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.counter = 0
        self.childs = {}

    def create_window(self, name):
        if name in self.childs:
            return False

        self.counter += 1
        t = tk.Toplevel(self)
        t.wm_title("Window[{}]".format(name))
        self.childs[name] = t
        t.protocol("WM_DELETE_WINDOW", lambda: self.delete_window(name))
        return True

    def delete_window(self, name):
        if not name in self.childs:
            return False

        t = self.childs[name]
        t.destroy()
        del self.childs[name]
        return True

class Checkbar(tk.Frame):
    def __init__(self, parent=None, picks=[], command=None, side=tk.LEFT, anchor=tk.W):
        tk.Frame.__init__(self, parent)

        self.command = command
        self.picks = []
        self.vars = []
        self.states = []
        for pick in picks:
            var = tk.IntVar()
            chk = tk.Checkbutton(self, text=pick, variable=var, command=lambda: self.on_state_change())
            chk.pack(side=side, anchor=anchor, expand=tk.YES)
            self.picks.append(pick)
            self.vars.append(var)
            self.states.append(0)

    def on_state_change(self):
        update = False
        for pick, var, state in zip(self.picks, self.vars, self.states):
            if var.get() != state:
                self.command(name=pick, trigger=var.get())
                update = True
        if update:
            self.states = self.state().values()

    def state(self):
        return dict(zip(self.picks, map((lambda var: var.get()), self.vars)))

def create_canvas(path, parent):
    try:
        sig, fs = audioload(path, sr=None, mono=False)
    except:
        return None

    if len(sig.shape) == 1:
        sig = np.reshape(sig, [1, len(sig)])

    parent.sig_info = {
        "sig": sig,
        "fs": fs
    }

    def refresh_figure(xlim=None, canvas=None):
        sig = parent.sig_info["sig"]
        fs = parent.sig_info["fs"]
        plt.gcf().clear()
        gc.collect()
        fig = plt.gcf()
        xmin = 0
        xmax = sig.shape[1]
        pics = 1

        if canvas and canvas.spectrum:
            pics += 1

        if canvas:
            canvas.fig = fig
            canvas.axes = {}

        if not xlim and canvas:
            xlim = canvas.xlim_stack[-1]

        if canvas or xlim:
            xmin = np.max([int(xlim[0] * fs), xmin])
            xmax = np.min([int(xlim[1] * fs), xmax])

        t = np.arange(0, sig.shape[1]) / fs
        for idx in range(sig.shape[0]):
            ax = plt.subplot(sig.shape[0] * pics, 1, idx * pics + 1)
            plt.plot(t[xmin:xmax], sig[idx, xmin:xmax])
            if canvas:
                canvas.axes["signal-{}".format(idx)] = ax
            if xlim:
                plt.xlim(xlim)

            offset = 0
            if canvas and canvas.spectrum:
                offset += 1
                ax = plt.subplot(sig.shape[0] * pics, 1, idx * pics + 1 + offset)
                freq_resp = fft(sig[idx, xmin:xmax], 512)
                freq_resp = freq_resp[:int(len(freq_resp)/2)]
                freq_resp = np.abs(freq_resp)
                ff = np.arange(0, len(freq_resp)+1) / (len(freq_resp)+1) * fs/2.
                ff = ff[:-1]
                if canvas.spectrum_log:
                    plt.semilogy(ff, freq_resp)
                else:
                    plt.plot(ff, freq_resp)
                canvas.axes["spectrum-{}".format(idx)] = ax
        
        fig.suptitle("fs: {} Hz, duration: {} secs".format(fs, sig.shape[1] * 1. / fs))

    def find_current_axes_name(iax, canvas):
        for name, ax in ({} if not hasattr(canvas, "axes") else canvas.axes).items():
            if ax == iax:
                return name
        return None

    def on_press(event, canvas):
        canvas.tictoc.toc()
        canvas.press_xpos = event.xdata

    def on_move(event, canvas):
        pass

    def on_release_on_signal_axes(event, canvas, interval):
        if interval > 150 and canvas.press_xpos != None and event.xdata != None:
            xdiff = canvas.press_xpos - event.xdata
            xlim = canvas.xlim_stack[-1]
            xlim[0] += xdiff
            xlim[1] += xdiff
            canvas.xlim_stack[-1] = xlim
            refresh_figure(xlim=xlim, canvas=canvas)
            canvas.draw()
            return

        if event.button == MouseButton.LEFT and event.xdata != None:
            xlim = canvas.xlim_stack[-1]
            xr = (xlim[1] - xlim[0]) * .7
            xmin = np.max([event.xdata - xr / 2, xlim[0]])
            xmax = np.min([xmin + xr, xlim[1]])
            canvas.xlim_stack.append([xmin, xmax])
            refresh_figure(xlim=canvas.xlim_stack[-1], canvas=canvas)
            canvas.draw()

        elif event.button == MouseButton.RIGHT:
            if len(canvas.xlim_stack) < 2:
                return
            del canvas.xlim_stack[-1]
            refresh_figure(xlim=canvas.xlim_stack[-1], canvas=canvas)
            canvas.draw()

    def on_release_on_spectrum_axes(event, canvas, interval):
        if interval > 150:
            return

        if event.button in [MouseButton.LEFT, MouseButton.RIGHT] and event.xdata != None:
            canvas.spectrum_log = not canvas.spectrum_log
            refresh_figure(canvas=canvas)
            canvas.draw()

    on_release_handle = {
        "signal": on_release_on_signal_axes,
        "spectrum": on_release_on_spectrum_axes,
    }

    def on_release(event, canvas):
        interval = canvas.tictoc.toc()
        ax_name = find_current_axes_name(event.inaxes, canvas)
        if not ax_name:
            return
        if not ax_name.split("-")[0] in on_release_handle:
            return

        on_release_handle[ax_name.split("-")[0]](event, canvas, interval)

    plt.gcf().clear()
    refresh_figure()
    canvas = FigureCanvasTkAgg(plt.gcf(), master=parent)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.draw()
    canvas.xlim_stack = [plt.gca().get_xlim()]
    canvas.tictoc = TicToc()
    canvas.tictoc.tic()
    canvas.spectrum = False
    canvas.spectrum_log = False
    refresh_figure(canvas=canvas)

    canvas.mpl_connect("button_press_event", lambda event: on_press(event, canvas))
    canvas.mpl_connect("button_release_event", lambda event: on_release(event, canvas))
    canvas.mpl_connect("motion_notify_event", lambda event: on_move(event, canvas))

    def config_changed(name, trigger):
        setattr(canvas, name, bool(trigger))
        refresh_figure(canvas=canvas)
        canvas.draw()

    funcs = Checkbar(parent=parent, picks=["spectrum"], command=config_changed)
    funcs.pack(side=tk.BOTTOM)

    return canvas

def run():
    root = tk.Tk()
    main = MainWindow(root)
    GLOBAL["root"] = root
    root.wm_title("Signal Analyzer")

    def _select_file():
        filenames = tk.filedialog.askopenfilenames(
            initialdir=os.getcwd(), title="Select files", filetypes=(("wav files", "*.wav"), ("mp3 files", "*.mp3")))

        if len(filenames) < 1:
            return
        if len(filenames) == 1 or True:
            name = filenames[0]
        else:
            name = "#{}".format(main.counter + 1)

        filenames = [n for n in filenames if type(n) == str and os.path.isfile(n)]
        if len(filenames) == 0 or not main.create_window(name):
            return

        win = main.childs[name]
        canvas = create_canvas(name, win)
        if not canvas:
            return
        
        win.canvas = canvas
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    button = tk.Button(master=root, text="Import audio", command=_select_file)
    button.pack(side=tk.TOP)

    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    button = tk.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tk.BOTTOM)

    main.pack(side="top", fill="both", expand=True)
    tk.mainloop()

if __name__ == "__main__":
    run()
