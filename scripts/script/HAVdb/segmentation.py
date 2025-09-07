import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib.patches import Rectangle




def get_mask_segmentation(time, signal, mask, common_time, value_bins=None, min_seg_length=0.15,min_seg_length2=0.15, min_seg_dist=0.1, min_seg_dist2=0.02):
    """
    Get the segmentation of the given signal based on the given mask.
    
    Args:
        time (np.ndarray): The time array.
        signal (np.ndarray): The signal array.
        mask (np.ndarray): The mask array.
        common_time (np.ndarray): The common time array.
        value_bins (list, optional): The list of possible values. Defaults to None.
        min_seg_length (float, optional): The minimum length of a segment. Defaults to 0.15.
        min_seg_dist (float, optional): The minimum distance between two segments. Defaults to 0.1.
        min_seg_dist2 (float, optional): The minimum distance between two segments. Defaults to 0.02.
        
    Returns:
        list: The segmentation of the given signal based on the given mask. [[t_start_seg, t_end_seg], [index_start_seg, index_end_seg], [value]]
    """
    
    #list segment of the data in a list of [t_start_seg_i, t_end_seg_i]
    seg = []
    start_seg = 0
    end_seg = 0
    for i in range(1,mask.shape[0]):
        if mask[i] == True and mask[i-1] == False:
            start_seg = i
        if mask[i] == False and mask[i-1] == True:
            end_seg = i
            seg.append([start_seg, end_seg])

    seg_time = [[time[seg[i][0]],time[seg[i][1]]] for i in range(len(seg))]

    #merge segments if they are too close (< 0.02s)
    i=0
    while i < len(seg_time)-1:
        if seg_time[i+1][0] - seg_time[i][1] < min_seg_dist2:
            seg_time[i][1] = seg_time[i+1][1]
            seg_time.pop(i+1)
        else:
            i = i+1
            
    #remove segments that are too short (< 0.2s)
    i=0
    while i < len(seg_time):
        if seg_time[i][1] - seg_time[i][0] < min_seg_length2:
            seg_time.pop(i)
        else:
            i = i+1
    #merge segments if they are too close (< 0.1s)      
    i=0
    while i < len(seg_time)-1:
        if seg_time[i+1][0] - seg_time[i][1] < min_seg_dist:
            seg_time[i][1] = seg_time[i+1][1]
            seg_time.pop(i+1)
        else:
            i = i+1
    #remove segments that are too short (< 0.2s)
    i=0
    while i < len(seg_time):
        if seg_time[i][1] - seg_time[i][0] < min_seg_length:
            seg_time.pop(i)
        else:
            i = i+1
            
    #compute the mean direction of each segment and get the closest angle
    seg_value = []
    for i in range(len(seg_time)):
        #transform the time to index
        index_start = np.where(time[:] >= seg_time[i][0])[0][0]
        index_end = np.where(time[:] >= seg_time[i][1])[0][0]
        #compute the mean direction
        mean_dir = np.mean(signal[index_start:index_end])
        #find the closest angle
        if value_bins is None:
            seg_value.append(mean_dir)
        else:
            closest_angle = min(value_bins, key=lambda x:abs(x-mean_dir))
            seg_value.append(closest_angle)

    segmentation = []
    for i in range(len(seg_time)):
        t_s = seg_time[i][0]
        t_e = seg_time[i][1]
        if t_s > common_time[0]:
            if t_s < common_time[-1]:
                c_t_s_ind = np.where(common_time >= t_s)[0][0]
            else:
                c_t_s_ind = common_time.shape[0]-1
        else:
            c_t_s_ind = 0
        if t_e < common_time[-1]:
            c_t_e_ind = np.where(common_time >= t_e)[0][0]
        else:
            c_t_e_ind = common_time.shape[0]-1
        if c_t_s_ind != c_t_e_ind:
            d = seg_value[i]
            index_d = value_bins.index(d) if value_bins is not None else d
            segmentation.append([t_s, t_e, d, index_d]) 

    return segmentation

def plot_segmentation(time, signal, segmentation, colors=None, mask=None, segmentation_count=0):
    """
    Plot the given signal and segmentation.

    Args:
        time (np.ndarray): The time array.
        signal (np.ndarray): The signal array.
        segmentation (list): The segmentation to plot. [[t_start_seg, t_end_seg], [index_start_seg, index_end_seg], [value]]
        colors (list): The colors to use for each value.
        mask (np.ndarray, optional): The mask array. Defaults to None.
        segmentation_count (int, optional): The segmentation count to use for the plot. Defaults to 0.
    """
    #plot direction and derivative
    plt.plot(time, signal)
    if mask is not None:
        plt.plot(time[mask], signal[mask], 'r*')
    #plot vertical lines for each segment
    for i in range(len(segmentation)):
        plt.axvline(x=segmentation[i][0], color='g')
        plt.axvline(x=segmentation[i][1], color='r')
        #plot a transparent rectangle for each segment depending on the value
        value = segmentation[i][2]
        if colors is None:
            #grey le
            col = "r"
        else:
            col = colors[value]
        plt.axvspan(segmentation[i][0], segmentation[i][1], facecolor=col, alpha=0.5)
        #add index of the segment
        if segmentation_count > 0:
            plt.text(segmentation[i][0], 0, str(i+segmentation_count), fontsize=12)
            plt.text(segmentation[i][0], 50, str(segmentation[i][1][0]), fontsize=9, rotation=90)
            plt.text(segmentation[i][1], -50, str(segmentation[i][1][1]), fontsize=9,rotation=90)
    plt.axvline(x=time[-1], color='k')
    
def interactive_segmentation(time, signal, segmentation, colors=None, mask=None, segmentation_count=0):
    """
    Plot the given signal and segmentation.

    Args:
        time (np.ndarray): The time array.
        signal (np.ndarray): The signal array.
        segmentation (list): The segmentation to plot. [[t_start_seg, t_end_seg], [index_start_seg, index_end_seg], [value]]
        colors (list): The colors to use for each value.
        mask (np.ndarray, optional): The mask array. Defaults to None.
        segmentation_count (int, optional): The segmentation count to use for the plot. Defaults to 0.
    """
    
    # Create a plot
    fig = plt.gcf()
    ax = plt.gca()
    ax.plot(time, signal)
    if mask is not None:
        ax.plot(time[mask], signal[mask], 'r*')
    ax.axvline(x=time[-1], color='k')
    # Create interactive vertical lines
    #print(segmentation) 
    if colors is None:
        colors = ["r"]
        seg = [[seg[0], seg[1], 0] for seg in segmentation]
    else:
        seg = [[seg[0], seg[1], seg[3]] for seg in segmentation]
    
    interactive_seg = InteractiveSegs(ax, seg, colors)
    return fig, ax, interactive_seg

def translate_seg_index(src_seg, src_time, dst_time):
    """
    Translate the segmentation index from the src_time to the dst_time.
    The src_seg contains the index of the src_time.
    the returned segmentation contains the index of the dst_time.

    Args:
        src_seg (list): The segmentation to translate.
        src_time (np.ndarray): The time array of the src_seg.
        dst_time (np.ndarray): The time array of the returned segmentation.
        
    Returns:
        list: The translated segmentation.
    """
    dst_seg = []
    for seg in src_seg:
        idx_s = seg[1][0]
        idx_e = seg[1][1]
        t_s = src_time[idx_s]
        if idx_e < src_time.shape[0]-1:
            
            t_e = src_time[idx_e]
            if t_e < dst_time[-1]:
                idx_s_dilated = np.where(dst_time >= t_s)[0][0]
                idx_e_dilated = np.where(dst_time >= t_e)[0][0]
                if idx_s_dilated != idx_e_dilated:
                    dst_seg.append([[t_s, t_e], [idx_s_dilated, idx_e_dilated], seg[2]])
    return dst_seg

class InteractiveSegs:
    def __init__(self, ax, segments, colors):
        self.ax = ax
        self.segments = []
        self.segments_lines = []
        self.colors = colors
        self.selected_line = None
        self.selected_segment = None
        self.type_segment = None
        for seg in segments:
            self.add_segment(seg)
        self.selected_color = None
        self.connect()
        self.add_button_new()
        self.add_button_delete()
        self.create_colors_buttons()

    def add_segment(self, seg):
        s_line = self.ax.axvline(seg[0], color="g", picker=5) 
        e_line = self.ax.axvline(seg[1], color="r", picker=5)
        span = self.ax.axvspan(seg[0], seg[1], facecolor=self.colors[seg[2]], alpha=0.5)
        rect = Rectangle((seg[0], 1 ), seg[1]-seg[0], 0.2, facecolor="k", alpha=0.1)
        rect.set_picker(True)
        self.segments_lines.append([s_line, e_line, span, rect])
        self.ax.add_patch(rect)
        self.segments.append(seg)
        self.selected_segment = len(self.segments)-1
        plt.draw()

    def connect(self):
        self.cid_pick = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'd':
            self.delete_segment(event)
        
        #test key "1" to "9" to change the color
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            color = int(event.key)-1
            self.selected_color = color
            if self.selected_segment is not None:
                self.segments_lines[self.selected_segment][2].set_facecolor(self.colors[self.selected_color])
                self.segments[self.selected_segment][2] = self.selected_color
                self.ax.figure.canvas.draw()

    def on_pick(self, event):
        for i, seg in enumerate(self.segments_lines):
            if event.artist in seg:
                if self.selected_segment is not None:
                    self.segments_lines[self.selected_segment][3].set(alpha=0.1, facecolor="k")
                self.selected_segment = i
                self.type_selected  = 'start' if event.artist == seg[0] else ('end' if event.artist == seg[1] else 'span')
                self.selected_line = seg[0] if event.artist == seg[0] else seg[1]
                self.segments_lines[self.selected_segment][3].set(alpha=1, facecolor="k")
                self.ax.figure.canvas.draw()
                break
    
    def on_release(self, event):
        self.selected_line = None
        self.type_selected = None

    def on_motion(self, event):
        if self.selected_line is None: return
        if event.xdata is not None:
            if self.type_selected == 'start':
                #move only if the xdata is smaller than the end line
                if event.xdata < self.segments_lines[self.selected_segment][1].get_xdata()[0]:
                    self.selected_line.set_xdata(event.xdata)
                    self.segments[self.selected_segment][0] = event.xdata
            elif self.type_selected == 'end':
                #move only if the xdata is bigger than the start line
                if event.xdata > self.segments_lines[self.selected_segment][0].get_xdata()[0]:
                    self.selected_line.set_xdata(event.xdata)
                    self.segments[self.selected_segment][1] = event.xdata
            elif self.type_selected == 'span':
                pass 
                
            #update the span
            px1 = self.segments_lines[self.selected_segment][0].get_xdata()[0]
            px2 = self.segments_lines[self.selected_segment][1].get_xdata()[0]
            self.segments_lines[self.selected_segment][2].set_xy([[px1, 0], [px1, 1], [px2, 1], [px2, 0]])
            self.segments_lines[self.selected_segment][3].set_width(px2-px1)
            self.segments_lines[self.selected_segment][3].set_xy([px1, 1])
            
            self.ax.figure.canvas.draw()
                    
    def add_button_new(self):
        ax_button_new = plt.axes([0.8, 0.01, 0.09, 0.05])
        self.button_new = Button(ax_button_new, 'Add')
        self.button_new.on_clicked(self.new_segment)
        
    def add_button_delete(self):
        ax_button_delete = plt.axes([0.9, 0.01, 0.09, 0.05])
        self.button_delete = Button(ax_button_delete, 'Delete')
        self.button_delete.on_clicked(self.delete_segment)
        
    def delete_segment(self, event):
        if self.selected_segment is not None:
            self.segments_lines[self.selected_segment][0].remove()
            self.segments_lines[self.selected_segment][1].remove()
            self.segments_lines[self.selected_segment][2].remove()
            self.segments_lines[self.selected_segment][3].remove()
            self.segments_lines.pop(self.selected_segment)
            self.segments.pop(self.selected_segment)
            self.selected_segment = None
            self.ax.figure.canvas.draw()
        
    def new_segment(self, event):
        if self.selected_segment is not None:
            seg = self.segments[self.selected_segment]
            x1 = seg[1]+0.5
            x2 = x1 + (seg[1]-seg[0])
            self.add_segment([x1, x2, seg[2]])
        else:
            self.add_segment([40, 45, 0])     
        
    def change_color(self, event):
        for i in range(len(self.colors)):
            if self.colors_buttons[i][1] == event.inaxes:
                color = i
                break
        self.selected_color = color
        if self.selected_segment is not None:
            self.segments_lines[self.selected_segment][2].set_facecolor(self.colors[self.selected_color])
            self.segments[self.selected_segment][2] = self.selected_color
            self.ax.figure.canvas.draw()
        
    def create_colors_buttons(self):
        #for each color in self.colors create a button with the color
        self.colors_buttons = []
        for i in range(len(self.colors)):
            ax_button = plt.axes([0.02 + (i*0.5)/len(self.colors), 0.01, 0.45/len(self.colors), 0.05])
            button = Button(ax_button, " ", color=self.colors[i])
            button.on_clicked(self.change_color)
            self.colors_buttons.append([button, ax_button])
        #set the first color as selected
        self.selected_color = self.colors[0]
        
    def get_segments(self):
        return self.segments
        
            
