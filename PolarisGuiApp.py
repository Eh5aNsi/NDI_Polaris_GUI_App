"""
 **Polaris Vega XT GUI App for Cadaver Experiments**

An advanced GUI for a Rigid Registration pipeline, plus real-time external tracking 
(Polaris Vega), and a 3D scene showing the phantom/CT.

-New feather: Works with Polaris Timing....


Author: [Ehsan Nasiri]
contact:[Ehsan.Nasiri@dartmouth.edu]

Date: Aug. 8th, 2025---> updated Dec. 10

"""

#############################################################################################################################
#############################################################################################################################

import sys
import argparse
import os
import time
import re
import threading
import subprocess
import socket
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import struct
import datetime
import serial
import serial.tools.list_ports
from sksurgerynditracker.nditracker import NDITracker
from scipy.spatial.transform import Rotation
import open3d as o3d
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit,
    QTextEdit, QFileDialog, QRadioButton, 
    QButtonGroup, QListWidget,
    QMessageBox,QGroupBox,QDialog,
    QScrollArea,QSizePolicy,QCheckBox  
)
from PyQt5.QtGui import ( QIntValidator,QIcon,
QPixmap, QFont
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches                
from matplotlib.collections import PatchCollection   
from mpl_toolkits.mplot3d import Axes3D
#import ipdb


# ------------------------------------------------------------------------------
# Directories & Utility Functions
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROM_DIR = BASE_DIR / 'ROM'            # all the .rom file for tools should be here
MODELS_DIR = BASE_DIR / 'model'
CALIB_DIR = BASE_DIR / 'calibration'  # .npy file for registrsation/calibration
TIP_CALIB = BASE_DIR / 'tips'         # all the .tip file for tools should be here
PHANTOM_DIR = MODELS_DIR / 'TORS'     # all the .stl or .vtk adn .fcsv etc realted to the phantom model should be here


# create them on startup  ----> ** put the .rom, .tip etc files within the following created folder first!! **
for d in (ROM_DIR, MODELS_DIR, CALIB_DIR,TIP_CALIB, PHANTOM_DIR):
    d.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------------------------------------------------------------------------#

def quat_to_mat(q):
    """Convert unit quaternion [q0,q1,q2,q3] to 3×3 rotation matrix."""
    q0, q1, q2, q3 = q
    
    R = np.array([
      [1 - 2*(q2**2+q3**2),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
      [2*(q1*q2 + q0*q3),     1 - 2*(q1**2+q3**2),   2*(q2*q3 - q0*q1)],
      [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2+q2**2)]
    ])
    
    return R

def polaris_crc16(cmd: str) -> str:
    data = cmd.encode('ascii')
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
   
    return f"{crc:04X}"


def isAlive(ip: str, timeout_ms: int) -> bool:
    try:
        out = subprocess.check_output(
            ['ping', '-n', '1', '-w', str(timeout_ms), ip],
            stderr=subprocess.DEVNULL
        ).decode('utf-8', errors='ignore')
        return '0% loss' in out
    except Exception:
        return False


def quat_rot(q_in: np.ndarray, v_in: np.ndarray) -> np.ndarray:
    q = q_in / np.linalg.norm(q_in)
    q0, q123 = q[0], q[1:]
    return (
        (q0**2 - np.dot(q123, q123)) * v_in
        + 2 * np.dot(q123, v_in) * q123
        + 2 * q0 * np.cross(q123, v_in)
    )


def pivot_calib_lsq(input_csv, std_dev_threshold=3.0, output_tip=None):
    # 1) load CSV
    df = pd.read_csv(input_csv)
    print(df.dtypes)          
    print(df.head(3))
    
    # single tool_id
    if df.tool_id.nunique() != 1:
        raise RuntimeError("pivot_calib_lsq: input must be a single tool")
    
    # 2) build full A, b
    A_blocks = []
    b_blocks = []
    for _, row in df.iterrows():
        t = row[['tx','ty','tz']].to_numpy().reshape(3,1)
        #q = row[['q0','q1','q2','q3']].to_numpy() # old
        q = np.array([row['q0'], row['q1'], row['q2'], row['q3']], dtype=float)
        R = quat_to_mat(q)
        A_blocks.append(np.hstack([R, -np.eye(3)]))
        b_blocks.append(-t)
        
    # A = np.vstack(A_blocks)
    # b = np.vstack(b_blocks)
    A = np.vstack(A_blocks).astype(float)
    b = np.vstack(b_blocks).astype(float)

    # initial LS
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    resids = (A @ x - b).reshape(-1,3).T  # 3×N

    # reject outliers
    mu = resids.mean(axis=1, keepdims=True)
    sigma = resids.std(axis=1, keepdims=True)
    z = np.abs((resids - mu) / sigma)
    mask = (z > std_dev_threshold).any(axis=0)  # length N
    keep = ~np.repeat(mask, 3).reshape(-1,1)
    
    # A2 = A[keep[:,0],:]
    # b2 = b[keep[:,0],:]
    A2 = A[keep[:,0], :].astype(float)
    b2 = b[keep[:,0], :].astype(float)

    # final LS
    x2, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    res2 = (A2 @ x2 - b2).reshape(-1,3)
    rmse = np.sqrt((res2**2).sum(axis=1).mean())
    tip_local = x2[:3,0]

    # write tip file
    if output_tip:
        with open(output_tip,'x') as f:
            f.write(','.join(f"{c:+.4f}" for c in tip_local))
            
    print(f"[pivot_calib_lsq] tip = {tip_local}   |   RMSE = {rmse:.3f} mm") # th RMSE for the tip calib.

    return tip_local, rmse


# ------------------------------------------------------------------------------
# Phantom Registration Class
# ------------------------------------------------------------------------------
class PhantomRegistration:
    def __init__(self):
        self.CHICKEN_FOOT_TOOLTIP_OFFSET = np.array([-304.5728, -0.3053, -0.1412, 1])
        self.SETTINGS_CF = {
            'tracker type': 'vega',
            'ip address': '169.254.7.250',
            'port': 8765,
            'romfiles': [str(ROM_DIR / 'medtronic_chicken_foot_960_556.rom')]
        }
        
        self.tracker_CF = NDITracker(self.SETTINGS_CF)
        self.tracker_CF.start_tracking()
        markers_ct_path = PHANTOM_DIR / 'Fiducials_CT.fcsv'
        self.markers_CT = pd.read_csv(str(markers_ct_path), comment='#', header=None).iloc[:, 1:4].to_numpy()
        self.markers_PO = []
        self.required_points = self.markers_CT.shape[0]

    def get_tooltip_data(self):
        port_handles, timestamps, frame_numbers, tracking, quality = self.tracker_CF.get_frame()
        tip = tracking[0] @ self.CHICKEN_FOOT_TOOLTIP_OFFSET
        return tip[:3]

    def collect_markers_PO_gui(self):
        dialog = QDialog()
        dialog.setWindowTitle('Collect Polaris Marker Points')
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(f'Collect {self.required_points} marker points.'))

        self.points_list = QListWidget()
        layout.addWidget(self.points_list)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        
        collect_btn = QPushButton('Collect Point')
        collect_btn.clicked.connect(self.on_collect_point)
        btn_layout.addWidget(collect_btn)

        finish_btn = QPushButton('Finish')
        finish_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(finish_btn)

        layout.addLayout(btn_layout)

        # block only dialog
        dialog.exec_()
        

    def on_collect_point(self):
        x, y, z = self.get_tooltip_data()
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            self.status_label.setText('Invalid reading. Please try again.')
        else:
            self.markers_PO.append([x, y, z])
            point_str = f'Point {len(self.markers_PO)}: ({x:.2f}, {y:.2f}, {z:.2f})'
            self.points_list.addItem(point_str)
            if len(self.markers_PO) >= self.required_points:
                self.status_label.setText('Done. Close GUI.')
            else:
                self.status_label.setText('Collected.')

    def compute_rigid_transform(self, P_PO: np.ndarray, P_CT: np.ndarray):
        centroid_CT = P_CT.mean(axis=0)
        centroid_PO = P_PO.mean(axis=0)
        H = (P_CT - centroid_CT).T @ (P_PO - centroid_PO)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        T = centroid_PO - R @ centroid_CT
        return R, T

    def run_registration(self):
        self.collect_markers_PO_gui()
        P_CT = self.markers_CT
        P_PO = np.array(self.markers_PO)
        R, T = self.compute_rigid_transform(P_PO, P_CT)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = T
        np.save(str(CALIB_DIR / 'POLARIS_to_CT_calib.npy'), M)
        
        return M

# ------------------------------------------------------------------------------
# Main GUI Class
# ------------------------------------------------------------------------------
class PolarisGUI(QMainWindow):
    def __init__(self, tool_defs=None):
        super().__init__()
        self.setWindowIcon(QIcon(str(BASE_DIR / 'logo.png'))) # logo icon for the console window
        self.setWindowTitle('Polaris GUI App')
        self.resize(1600, 1000)
        self._init_data()
        self._init_ui()
         
        if tool_defs:
            for idx, (rom, tip) in enumerate(tool_defs[:3]):
                # populate the idx-th row
                self.toolDefFiles[idx] = [rom, 'D']
                if tip:
                    self.tipCalFiles[idx]   = tip
                    self.tipCalData[idx, :] = np.loadtxt(tip, delimiter=',')
                # Update UI
                self.toolGrid.itemAtPosition(idx,1).widget().setText(os.path.basename(rom))
                self.toolGrid.itemAtPosition(idx,3).widget().setText(os.path.basename(tip) if tip else '')

        self.updateToolDefDisplay()
        
        self._init_plots()
        self.resetToolStatusIndicators()
        self.captureTimer = QTimer(self)
        self.captureTimer.timeout.connect(self._captureIteration)
    

    def _init_data(self):
        self.toolDefFiles = [['', '']] * 9
        self.tipCalFiles = [''] * 9
        self.tipCalData = np.zeros((9, 3))
        self.doUseTipCal = False
        self.outputFilePath = ''
        self.fidDataOut = None
        self.fidSerial = None
        self.ndiTracker = None
        self.useEthernet = False
        self.gx_cmd_str = ''   # for serial communication & ehternet communicaton without NDI SDK!
        self.pstat_cmd_str = ''
        self.BASE_TOOL_CHAR = 64
        self.toolsUsed = []
        self.gx_transform_map = [6,7,8,10,11,12,14,15,16] # for serial communication & ehternet communicaton without NDI SDK!
        self.endTrackingFlag = False
        self.DEBUG_MODE = False
        self.send_video_cmd = False
        self.previewFlag = False
        self.baseUnixTimestamp = 0.0
        self.registration_matrix = None
        self.pointHandles = []
        self.FOV = {
            'NEAR_Z': 950.0,
            'FAR_Z': 950.0 + 2050.0,         
            'TOP_BREAK_Z': 950.0 + 1532.0,

            'W_NEAR': 480.0 / 2.0,
            'H_NEAR': 448.0 / 2.0,
            'W_FAR_FRONT': 1856.0 / 2.0,
            'H_FAR_SIDE': 1470.0 / 2.0,

            'TOP_ANG_STEEP':   np.deg2rad(29.66),
            'TOP_ANG_SHALLOW': np.deg2rad(13.66),
            'BORDER_TOL': 10.0
            }
        self.lastFinishedCSV = ''   # path of the most recent *finished* CSV
        #-----------------------------------------------------------------------------
        #NDI tracker clock (Polaris counter derived from PTP, 60 Hz)
        # Auto-rate detection with a temporary default  
        self.DEFAULT_POLARIS_HZ   = 400.0   # sensible for Vega XT
        self.POLARIS_HZ           = None    # locked/settled rate
        self.ndi_frame0           = None    # baseline frame #
        self.host_t0              = None    # host time at baseline
        self._rate_est            = None    # running estimate 
        self._rate_samples        = 0
        self._using_default_rate  = True
        #-----------------------------------------------------------------------------------
        # --- HyperDeck config ---
        self.HD_LEFT_IP   = '192.168.10.50'
        self.HD_RIGHT_IP  = '192.168.10.60'
        self.HD_PORT      = 9993
        self.HD_TIMEOUT_S = 2.0
        self.hd_left_sock  = None
        self.hd_right_sock = None
        self.video_running = False   # track current recording state
        # Pause/resume state for continuous capture
        self._paused = False

   
    def _init_ui(self):
        w = QWidget()
        #self.setCentralWidget(w) #   ---> uncoment for HQ 4K display sytem only!!
        
        scroll = QScrollArea()  # ---> uncomment the following 4 lines are for ordinary laptp display other than 4k display!!!
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        self.setCentralWidget(scroll)
        
        
        main_lay = QHBoxLayout(w)

        # Connection type selection
        conn_label = QLabel('Connection Type:')
        self.connType = QComboBox()
        
        self.connType.addItems(['Ethernet','Serial']) # default is Ehternet
        self.connType.currentIndexChanged.connect(self.on_connType_change)
        
        # COM port
        port_label = QLabel('COM Port:')
        self.comport = QComboBox()
        
        self.comport.addItem('Auto')
        for port in serial.tools.list_ports.comports():
            self.comport.addItem(port.device)
        
        # IP address & port
        ip_label = QLabel('IP Address:')
        self.ipAddress = QLineEdit('169.254.7.250')
        portnum_label = QLabel('IP Port:')
        self.ipPort = QLineEdit('8765')
        self.ipPort.setValidator(QIntValidator(1,65535))

        # Left panel
        ctrl = QWidget()
        ctrl_l = QVBoxLayout(ctrl)
        
        self.ptd_rom_btn   = QPushButton("Select .rom")
        self.ptd_rom_clear = QPushButton("Clear")
        self.ptd_tip_btn   = QPushButton("Select .tip")
        self.ptd_tip_clear = QPushButton("Clear")
        
        self.ptd_rom_path  = QLineEdit()
        self.ptd_rom_path.setReadOnly(True)
        self.ptd_tip_path  = QLineEdit()
        self.ptd_tip_path.setReadOnly(True)
        
        self.toolid = QComboBox()
        for i in range(1, 10):
            self.toolid.addItem(chr(64 + i))
        
        self.rbDynamic      = QRadioButton("Dynamic") # default
        self.rbStatic       = QRadioButton("Static")
        self.rbDynamic.setChecked(True)

        # --- Tool Definitions group ---
        ptd = QGroupBox("Passive Tool Definitions")
        ptd_l = QGridLayout(ptd)
        
        # Tool ID + Dynamic/Static
        ptd_l.addWidget(QLabel("Tool ID:"),  0, 0)
        ptd_l.addWidget(self.toolid,         0, 1)
        ptd_l.addWidget(self.rbDynamic,      1, 0)
        ptd_l.addWidget(self.rbStatic,       1, 1)
        ptd_l.addWidget(self.ptd_rom_btn,    2, 0)
        ptd_l.addWidget(self.ptd_rom_clear,  2, 1)
        ptd_l.addWidget(self.ptd_rom_path,   2, 2)
        ptd_l.addWidget(self.ptd_tip_btn,    3, 0)
        ptd_l.addWidget(self.ptd_tip_clear,  3, 1)
        ptd_l.addWidget(self.ptd_tip_path,   3, 2)
        
        ctrl_l.insertWidget(0, ptd)  
        ptd.hide()

        main_lay.addWidget(ctrl, 0)

        ctrl_l.addWidget(conn_label)
        ctrl_l.addWidget(self.connType)
        ctrl_l.addWidget(port_label)
        ctrl_l.addWidget(self.comport)
        ctrl_l.addWidget(ip_label)
        ctrl_l.addWidget(self.ipAddress)
        ctrl_l.addWidget(portnum_label)
        ctrl_l.addWidget(self.ipPort)

        # # Left panel
        # ctrl = QWidget()
        # ctrl_l = QVBoxLayout(ctrl)
        # main_lay.addWidget(ctrl, 0)

        # # COM port
        # ctrl_l.addWidget(QLabel('COM Port:'))
        # self.comport = QComboBox()
        # self.comport.addItem('Auto')
        # for port in serial.tools.list_ports.comports():
        #     self.comport.addItem(port.device)
        # ctrl_l.addWidget(self.comport)

        # Mode radio buttons
        ctrl_l.addWidget(QLabel('Mode:'))
        self.rbtrack = QRadioButton('Tracking')
        self.rbid = QRadioButton('Tool ID')
        self.rbtrack.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.rbtrack)
        mode_group.addButton(self.rbid)
        ctrl_l.addWidget(self.rbtrack)
        ctrl_l.addWidget(self.rbid)

        # Number of markers
        ctrl_l.addWidget(QLabel('Number of Markers:'))
        self.nummarkers = QComboBox()
        
        for i in range(1, 17):
            self.nummarkers.addItem(str(i))
        ctrl_l.addWidget(self.nummarkers)

        # Tool ID selection
        ctrl_l.addWidget(QLabel('Tool ID:'))
        
        ctrl_l.addWidget(self.toolid)

        # Tool definition file
        self.tooldeffile = QLineEdit('No tool definition file selected!')
        self.tooldeffile.setReadOnly(True)
        ctrl_l.addWidget(self.tooldeffile)
        tdf_layout = QHBoxLayout()
        
        self.tooldefbutton = QPushButton('Select .rom')
        self.tooldefclearbutton = QPushButton('Clear')
        tdf_layout.addWidget(self.tooldefbutton)
        tdf_layout.addWidget(self.tooldefclearbutton)
        ctrl_l.addLayout(tdf_layout)

        # Tip calibration file
        self.tipcalfile = QLineEdit('No tip calibration file selected!')
        self.tipcalfile.setReadOnly(True)
        ctrl_l.addWidget(self.tipcalfile)
        
        tcf_layout = QHBoxLayout()
        self.tipcalbutton = QPushButton('Select .tip')
        self.tipcalclearbutton = QPushButton('Clear')
        tcf_layout.addWidget(self.tipcalbutton)
        tcf_layout.addWidget(self.tipcalclearbutton)
        ctrl_l.addLayout(tcf_layout)

        # Output file
        ctrl_l.addWidget(QLabel('Output File:'))
        self.outputfile = QLineEdit('trial001.csv')
        self.outputfile.textChanged.connect(self.on_outputfile_text_changed)  #  for manual chnanging name
        ctrl_l.addWidget(self.outputfile)
        
        of_layout = QHBoxLayout()
        self.outputfileselectbutton = QPushButton('Browse')
        self.outputfileclearbutton = QPushButton('Clear')
        of_layout.addWidget(self.outputfileselectbutton)
        of_layout.addWidget(self.outputfileclearbutton)
        ctrl_l.addLayout(of_layout)

        ###
        # Single Cap append option 
        self.cbAppendSingles = QCheckBox("Single Cap → append to this file")
        self.cbAppendSingles.setChecked(False)
        ctrl_l.addWidget(self.cbAppendSingles)
        ###

        # Capture note
        ctrl_l.addWidget(QLabel('Capture Note:'))
        self.capturenote = QLineEdit()
        ctrl_l.addWidget(self.capturenote)
        
        # keep capture‐note enabled only in Tracking mode
        self.rbtrack.toggled.connect(self._update_capture_note_enable)
        # initialize its state
        self._update_capture_note_enable()

        # Valid frame count
        ctrl_l.addWidget(QLabel('Valid Frame Count:'))
        self.frameCount = QLineEdit('0')
        self.frameCount.setReadOnly(True)
        ctrl_l.addWidget(self.frameCount)

        # Control buttons
        btn_names = [
            'Connect', 'Disconnect', 'Single Cap', 'Start Cap', 'Start Cap + Video',
            'Stop', 'Pause', 'Preview', 'Calibrate Tip', 'Generate ROM', 'Register Phantom'
        ]
        
        self.buttons = {}
        for name in btn_names:
            btn = QPushButton(name)
            ctrl_l.addWidget(btn)
            self.buttons[name] = btn

        # Status indicators grid
        status_grid = QGridLayout()
        labels = list('ABCDEFGHI')
        
        self.status_labels = {}  ## added newly......

        for idx, lab in enumerate(labels):
            lbl_widget = QLabel(lab)
            lbl_widget.setFixedSize(30, 30)
            lbl_widget.setStyleSheet('background-color: lightgray; border: 1px solid black;')
            
            status_grid.addWidget(lbl_widget, idx//3, idx%3)
            # self.status_labels = {lab: lbl_widget}
            self.status_labels[lab] = lbl_widget ## added newly.....
            
        ctrl_l.addLayout(status_grid)
        
        # --- Legend for status colors ---
        legend_layout = QHBoxLayout()
        for name, color in [
         ("Tracking",        "lightgreen"),
         ("Partially Out",   "yellow"),
         ("Out of Vol.",   "red"),
         #("Too Few Markers", "cyan"),
         ("Not Loaded",      "lightgray"),
        ]:
            
         lbl = QLabel(name)
         lbl.setStyleSheet(
         f"background-color: {color};"
         " border: 1px solid black;"
         " padding: 4px;"
         )
         
         legend_layout.addWidget(lbl)
        ctrl_l.addLayout(legend_layout)

        # Console log
        ctrl_l.addWidget(QLabel('Console:'))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        ctrl_l.addWidget(self.console, 1)
        self.console.setStyleSheet("""
        QTextEdit {
        color:     #b5f4ff;
        }
        QTextEdit::viewport {
        background-color: #1e1e1e;
        }
        """)
        
        ctrl_l.addStretch()

        # Plot canvases
        plot_layout = QVBoxLayout()
        
        main_lay.addLayout(plot_layout, 1)
        self.canvas_front = FigureCanvas(Figure(figsize=(4,4)))
        self.canvas_top = FigureCanvas(Figure(figsize=(4,4)))
        self.canvas_side = FigureCanvas(Figure(figsize=(4,4)))
        self.canvas_ct = FigureCanvas(Figure(figsize=(5,5)))
        
        plot_layout.addWidget(QLabel('Front View', alignment=Qt.AlignCenter))
        plot_layout.addWidget(self.canvas_front)
        plot_layout.addWidget(QLabel('Top View', alignment=Qt.AlignCenter))
        plot_layout.addWidget(self.canvas_top)
        
        plot_layout.addWidget(QLabel('Side View', alignment=Qt.AlignCenter))
        plot_layout.addWidget(self.canvas_side)
        plot_layout.addWidget(QLabel('3D CT View', alignment=Qt.AlignCenter))
        plot_layout.addWidget(self.canvas_ct)

        # Connect signals
        self.buttons['Connect'].clicked.connect(self.on_connect)
        self.buttons['Disconnect'].clicked.connect(self.on_disconnect)
        self.buttons['Single Cap'].clicked.connect(self.on_singlecap)
        self.buttons['Start Cap'].clicked.connect(self.on_startcap)
        self.buttons['Start Cap + Video'].clicked.connect(self.on_startcapvid)
        self.buttons['Stop'].clicked.connect(self.on_stopcap)
        self.buttons['Pause'].clicked.connect(self.on_pausecap)  # NEW..
        self.buttons['Preview'].clicked.connect(self.on_preview)
        self.buttons['Calibrate Tip'].clicked.connect(self.on_tipcalibrate)
        self.buttons['Generate ROM'].clicked.connect(self.on_generateROM)
        self.buttons['Register Phantom'].clicked.connect(self.on_register_phantom)
        
        self.tooldefbutton.clicked.connect(self.on_tooldef_select)
        self.tooldefclearbutton.clicked.connect(self.on_tooldef_clear)
        self.tipcalbutton.clicked.connect(self.on_tipcal_select)
        self.tipcalclearbutton.clicked.connect(self.on_tipcal_clear)
        self.outputfileselectbutton.clicked.connect(self.on_outputfile_select)
        self.outputfileclearbutton.clicked.connect(self.on_outputfile_clear)
        self.toolid.currentIndexChanged.connect(self.updateToolDefDisplay)
        
        # ROM buttons:
        ptd.findChildren(QPushButton)[0].clicked.connect(self.on_tooldef_select)
        ptd.findChildren(QPushButton)[1].clicked.connect(self.on_tooldef_clear)
        
        # TIP buttons:
        ptd.findChildren(QPushButton)[2].clicked.connect(self.on_tipcal_select)
        ptd.findChildren(QPushButton)[3].clicked.connect(self.on_tipcal_clear)
        
        self.ptd_rom_btn  .clicked.connect(self.on_tooldef_select)
        self.ptd_rom_clear.clicked.connect(self.on_tooldef_clear)
        self.ptd_tip_btn  .clicked.connect(self.on_tipcal_select)
        self.ptd_tip_clear.clicked.connect(self.on_tipcal_clear)

        mode_group.buttonClicked.connect(self.on_mode_change)

        # Disable non-connect buttons initially
        for name, btn in self.buttons.items():
            #if name != 'Connect':
             if name not in ('Connect', 'Register Phantom'): # can either be register the phantom or connect and capture...
                btn.setEnabled(False)
                
                
    def _update_capture_note_enable(self):
        """Enable the note field only when 'Tracking' mode is selected."""
        self.capturenote.setEnabled(self.rbtrack.isChecked())
        
        
    def _open_csv_if_needed(self):
        """
        Open CSV handle if not open.
        - If 'Single Cap → append...' is checked and file exists, open in append.
        - Otherwise open in write mode.
        - Write header only if file is new/empty.
        """
        if self.fidDataOut is not None:
            return

        # Resolve desired filename
        out = (self.outputFilePath or self.outputfile.text().strip() or "").strip()
        if not out:
            existing = glob.glob('trial*.csv')
            idxs = [int(f[5:-4]) for f in existing if f[5:-4].isdigit()]
            out = f"trial{(max(idxs)+1 if idxs else 1):03d}.csv"
        if not out.lower().endswith('.csv'):
            out += '.csv'

        # Decide append vs write
        append_mode = bool(self.cbAppendSingles.isChecked() and Path(out).exists())

        self.outputFilePath = out
        self.updateOutputFilePath()

        mode = 'a' if append_mode else 'w'
        self.fidDataOut = open(self.outputFilePath, mode, encoding='utf-8')

        # header only if new/empty
        need_header = True
        if append_mode:
            try:
                need_header = (Path(self.outputFilePath).stat().st_size == 0)
            except Exception:
                need_header = True

        if need_header:
            self.fidDataOut.write(
                'polaris_time,unix_time,tool_id,q0,q1,q2,q3,tx,ty,tz,'
                'tx_tip,ty_tip,tz_tip,reg_err,capture_note\n'
            )

    ###
    def _hd_send(self, sock: socket.socket, cmd: str, expect_ok=True) -> str:
        """
        Send one HyperDeck command and read one line reply.
        Protocol wants CRLF over TCP. Expect replies like '200 ok'.
        """
        if sock is None:
            return ""
        msg = (cmd + "\r\n").encode("ascii")
        sock.sendall(msg)
        sock.settimeout(self.HD_TIMEOUT_S)
        data = sock.recv(1024)  # single line is fine
        resp = data.decode("ascii", errors="ignore").strip()
        if expect_ok and not resp.lower().startswith("200"):
            self.log(f"[HyperDeck] Unexpected reply to '{cmd}': {resp}")
        else:
            self.log(f"[HyperDeck] {cmd} -> {resp}")
        return resp

    def _hd_open_one(self, ip: str) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.HD_TIMEOUT_S)
        s.connect((ip, self.HD_PORT))
        # quick software ping
        self._hd_send(s, "ping")
        # enable remote + select slot 1
        self._hd_send(s, "remote: enable: true")
        self._hd_send(s, "slot select: slot id: 1")
        return s

    def _video_start(self):
        """
        Open both HyperDeck sockets and start recording on each.
        Only runs once per session; safe to call if already recording.
        """
        if self.video_running:
            return
        # Optional network check...
        try:
            self.hd_left_sock  = self._hd_open_one(self.HD_LEFT_IP)
        except Exception as e:
            self.log(f"[HyperDeck] Left open failed: {e}")
            self.hd_left_sock = None
        try:
            self.hd_right_sock = self._hd_open_one(self.HD_RIGHT_IP)
        except Exception as e:
            self.log(f"[HyperDeck] Right open failed: {e}")
            self.hd_right_sock = None

        # start recording (any side that is connected)
        if self.hd_left_sock:  self._hd_send(self.hd_left_sock,  "record")
        if self.hd_right_sock: self._hd_send(self.hd_right_sock, "record")

        self.video_running = bool(self.hd_left_sock or self.hd_right_sock)
        if self.video_running:
            self.log("[HyperDeck] Recording started.")
        else:
            self.log("[HyperDeck] No recorders reachable; video not started.")

    def _video_stop(self):
        """
        Stop recording and close sockets.
        """
        if self.hd_left_sock:
            try: self._hd_send(self.hd_left_sock, "stop")
            except: pass
            try: self.hd_left_sock.close()
            except: pass
            self.hd_left_sock = None

        if self.hd_right_sock:
            try: self._hd_send(self.hd_right_sock, "stop")
            except: pass
            try: self.hd_right_sock.close()
            except: pass
            self.hd_right_sock = None

        if self.video_running:
            self.log("[HyperDeck] Recording stopped.")
        self.video_running = False
        ###


    def _init_plots(self):
        theta = np.linspace(0, 2*np.pi, 400)
        r = 500 
        # # Front view
        # axf = self.canvas_front.figure.subplots()
        # axf.set_aspect('equal'); axf.grid(True)
        # axf.set_xlim(-750,750); axf.set_ylim(-750,750)
        # axf.plot(r*np.cos(theta), r*np.sin(theta), 'k-', lw=1.6)
        # axf.plot([-750,750],[-650,-650],':', color='0.7', lw=6)
        # self.ax_front = axf
        # Front view (true FOV)
        
        axf = self.canvas_front.figure.subplots()
        
        axf.set_aspect('equal'); axf.grid(True)
        axf.set_xlim(-2000,2000); axf.set_ylim(-2000,2000)

        # near, far = 950.0, 950.0 + 2400.0
        # w_near, h_near = 1566/2, 1312/2
        # w_far,  h_far  = 1856/2, 1470/2
        # ----------------------------- ➊ FOV CONSTANTS -----------------------------
       
        # ───────────────────────────────────────────────────────────────────────
        #  Vega-XT geometry  (numbers from the data sheet...)
        # ───────────────────────────────────────────────────────────────────────
        #                FRONT              TOP                SIDE
        NEAR_Z   = 950.0                     # distance to the near plane
        FAR_Z    =  2600.0                    # NEAR   + 3000 mm

        W_NEAR   = 480.0   / 2               # rectangle 480 mm × 448 mm
        H_NEAR   = 448.0   / 2

        W_FAR_F  = 1856.0 / 2                # front view width @ far
        H_FAR_S  = 1470.0 / 2                # side  view height @ far

        # top-view half-widths – has TWO linear slopes...:
        #   • first (steeper) until the “extended” break-plane  (29.66°)
        #   • then a shallower pyramid to FAR_Z (13.66°)
        BREAK_Z  =  + 1532.0           # 950 + 1532 = 2482 mm

        ANG_STEEP  = np.deg2rad(29.66) #/ 2   # half-angle   (two-sided)
        ANG_SHALLOW= np.deg2rad(13.66) #/ 2

        Y_BREAK = W_NEAR + (BREAK_Z - NEAR_Z) * np.tan(ANG_STEEP)
        Y_FAR   = Y_BREAK + (FAR_Z   - BREAK_Z) * np.tan(ANG_SHALLOW)

        # -----------------------------------------------------------------------
        #  1. FRONT VIEW  ────────────────────────────────────────────────────────
        # -----------------------------------------------------------------------
        axf = self.canvas_front.figure.subplots()
        axf.set_aspect('equal') ; axf.grid(True)
        axf.set_xlim(-W_FAR_F*1.1,  W_FAR_F*1.1)
        axf.set_ylim(-H_FAR_S*1.1,  H_FAR_S*1.1)

        # outer rectangle (far plane) + inner (near plane)
        outer = patches.Rectangle((-W_FAR_F, -H_FAR_S),
                                2*W_FAR_F, 2*H_FAR_S,
                                fill=False, lw=1.6)
        inner = patches.Rectangle((-W_NEAR, -H_NEAR),
                                2*W_NEAR, 2*H_NEAR,
                                fill=False, lw=1.6)
        axf.add_patch(outer);  axf.add_patch(inner)

        # pyramid edges
        for sx in (-1, 1):
            for sy in (-1, 1):
                axf.plot([sx*W_NEAR, sx*W_FAR_F],
                        [sy*H_NEAR, sy*H_FAR_S], color='0.3')
        self.ax_front = axf


        # -----------------------------------------------------------------------
        # 2. TOP VIEW  (plan view, Z → –X plot,  Y → –Y plot)
        # -----------------------------------------------------------------------
        # ─────────────────────────────────────────────────────────────
        NEAR_Z   = 950.0                         # mm
        BREAK_Z  = NEAR_Z + 1532.0               # 950 + 1532 = 2482 mm
        FAR_Z    = NEAR_Z + 1650.0               # 950 + 3000 = 3950 mm

        W_NEAR   = 480.0 / 2                     # near half-width (mm)

        # Half-angles from centerline to one side
        ANG_STEEP   = np.deg2rad(29.66)          # first section
        ANG_SHALLOW = np.deg2rad(13.66)          # second section

        # Lateral half-widths produced by each section
        Y_BREAK = W_NEAR + (BREAK_Z - NEAR_Z) * np.tan(ANG_STEEP)
        Y_FAR   = Y_BREAK + (FAR_Z   - BREAK_Z) * np.tan(ANG_SHALLOW)

        axt = self.canvas_top.figure.subplots()
        axt.set_aspect('equal'); axt.grid(True)
        axt.set_xlim(-FAR_Z * 0.05, FAR_Z * 1.05)
        axt.set_ylim(-Y_FAR * 1.1, Y_FAR * 1.1)

        # Polygon: near vertical → steep slope to BREAK_Z → shallow slope to FAR_Z
        verts_top = [
            (NEAR_Z,  -W_NEAR),
            (NEAR_Z,   W_NEAR),
            (BREAK_Z,  Y_BREAK),
            (FAR_Z,    Y_FAR),
            (FAR_Z,   -Y_FAR),
            (BREAK_Z, -Y_BREAK),
        ]

        poly_top = patches.Polygon(verts_top, closed=True, lw=1.6,
                                facecolor='skyblue', alpha=0.07)
        axt.add_patch(poly_top)
        axt.add_patch(
            patches.Polygon(verts_top, closed=True, fill=False, lw=2.0, edgecolor='k')
        )
        self.ax_top = axt


        # -----------------------------------------------------------------------
        # 3. SIDE VIEW  (elevation, Z → –X plot,  X → –Y plot)
        # -----------------------------------------------------------------------
        axs = self.canvas_side.figure.subplots()
        axs.set_aspect('equal')
        axs.grid(True)

        axs.set_xlim(-FAR_Z * 0.05, FAR_Z * 1.05)
        axs.set_ylim(-H_FAR_S * 1.1, H_FAR_S * 1.1)

        # Lengths 
        near_z = NEAR_Z
        near_h = H_NEAR
        slope_length = 500  # distance from near vertical to start of far vertical
        angle_deg = 16.58
        angle_rad = np.deg2rad(angle_deg)

        # Compute height increase over slope section
        delta_h = np.tan(angle_rad) * slope_length

        # Z positions
        slope_end_z = near_z + slope_length
        far_z = FAR_Z

        # Heights
        slope_top_h = near_h + delta_h
        slope_bottom_h = -near_h - delta_h

        # Polygon vertices 
        verts_side = [
            (near_z, -near_h),                  # bottom near
            (near_z,  near_h),                  # top near
            (slope_end_z, slope_top_h),         # top slope end
            (far_z,   H_FAR_S),                  # top far
            (far_z,  -H_FAR_S),                  # bottom far
            (slope_end_z, slope_bottom_h)       # bottom slope end
        ]

        poly_side = patches.Polygon(verts_side, closed=True, lw=1.6,
                                    facecolor='skyblue', alpha=0.07)
        axs.add_patch(poly_side)
        axs.add_patch(
            patches.Polygon(verts_side, closed=True,
                            fill=False,
                            lw=2.0, edgecolor='k'))
        self.ax_side = axs


        
        # Create 9 text‐handles (A…I), 
        self.pointHandles = []
        for idx in range(9):
            lbl = chr(65 + idx)  # 'A','B',…, 'I'
            
            front = axf.text(
                np.nan, np.nan, lbl,
                fontsize=18, fontweight='bold',
                ha='center', va='center', color='green'
            )
            
            top = axt.text(
                np.nan, np.nan, lbl,
                fontsize=18, fontweight='bold',
                ha='center', va='center', color='green'
            )
            
            side = axs.text(
                np.nan, np.nan, lbl,
                fontsize=18, fontweight='bold',
                ha='center', va='center', color='green'
            )
            
            self.pointHandles.append({
                'front': front,
                'top':   top,
                'side':  side
            })

        # CT 3D plot
        axct = self.canvas_ct.figure.add_subplot(111, projection='3d')
        axct.set_title('CT & Tool in CT Frame')
        axct.set_xlabel('X'); axct.set_ylabel('Y'); axct.set_zlabel('Z')
        
        fcsv = PHANTOM_DIR / 'Fiducials_CT.fcsv'
        if fcsv.exists():
            P_CT = pd.read_csv(str(fcsv), comment='#', header=None).iloc[:,1:4].to_numpy()
            axct.scatter(P_CT[:,0],P_CT[:,1],P_CT[:,2], marker='o', s=50)
        self.ax_ct = axct
        
        self.tool_ct_handle = None
        
        # Draw all canvases
        self.canvas_front.draw()
        self.canvas_top.draw()
        self.canvas_side.draw()
        self.canvas_ct.draw()

    # Logging helper
    def log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.console.append(f'[{ts}] {msg}')

    # UI state helpers
    def resetToolStatusIndicators(self):
        for lbl in self.status_labels.values():
            lbl.setStyleSheet('background-color: lightgray; border: 1px solid black;')
            

    def setToolStatusIndicator(self, toolIdx: int, statusCode: int):
        colors = {0: 'lightgreen', 1: 'yellow', 2: 'red', 3: 'orange', 4: 'lightgray'}
        lbl = self.status_labels[chr(64+toolIdx)]
        lbl.setStyleSheet(f'background-color: {colors[statusCode]}; border: 1px solid black;')

    # def updateToolDefDisplay(self): #(old)
    #     idx = self.toolid.currentIndex()
    #     file, typ = self.toolDefFiles[idx]
    #     self.tooldeffile.setText(os.path.basename(file) if file else 'No tool definition file selected!')
    #     tip = self.tipCalFiles[idx]
    #     self.tipcalfile.setText(os.path.basename(tip) if tip else 'No tip calibration file selected!')
    #     self.doUseTipCal = any(bool(x) for x in self.tipCalFiles)
    
    def updateToolDefDisplay(self):
        idx = self.toolid.currentIndex()
        rom, _ = self.toolDefFiles[idx]
        tip = self.tipCalFiles[idx]

        # update our PTD group line-edits
        self.ptd_rom_path.setText(os.path.basename(rom) if rom else "")
        self.ptd_tip_path.setText(os.path.basename(tip) if tip else "")
        
        # Update the middle‐of‐screen tooldeffile for rom  & tip
        if rom:
            self.tooldeffile.setText(os.path.basename(rom))
        else:
            self.tooldeffile.setText('No tool definition file selected!')
            
        if tip:
            self.tipcalfile.setText(os.path.basename(tip))
        else:
            self.tipcalfile.setText('No tip calibration file selected!')

        # keep the enable/disable logic
        self.doUseTipCal = any(bool(x) for x in self.tipCalFiles)


    # def updateOutputFilePath(self):
    #     if not self.outputFilePath:
    #         self.outputfile.setText('Test_01.csv')
    #     else:
    #         self.outputfile.setText(os.path.basename(self.outputFilePath))
    
    def updateOutputFilePath(self):
        # only push → UI when *self* ---> fro manual changing name file
        if self.outputFilePath:
            self.outputfile.setText(os.path.basename(self.outputFilePath))

    def on_outputfile_text_changed(self, text: str):
        """Keep internal path in sync with whatever the user types."""
        self.outputFilePath = text.strip()


    
    
    def on_mode_change(self):
        """Toggle number-of-markers field based on mode RadioButtons."""
        
        if self.rbtrack.isChecked():
            self.nummarkers.setEnabled(False)
        else:
            self.nummarkers.setEnabled(True)

    def on_connType_change(self, index):
        """Enable/disable Serial vs Ethernet fields based on selection."""
        
        useEth = (self.connType.currentText() == 'Ethernet')
        self.comport.setEnabled(not useEth)
        self.ipAddress.setEnabled(useEth)
        self.ipPort.setEnabled(useEth)

    # Callback implementations
    def on_register_phantom(self):
        reg = PhantomRegistration()
        M = reg.C()
        self.registration_matrix = M
        self.statusBar().showMessage('Phantom registered successfully.')
        
        # Refresh CT plot
        self.ax_ct.cla()
        self._init_plots()

    def on_tipcalibrate(self):
        csv_for_tip = (self.lastFinishedCSV or self.outputfile.text().strip())
        if not csv_for_tip.lower().endswith('.csv') or not Path(csv_for_tip).exists():
            QMessageBox.warning(self, 'Warning', 'No finished CSV trial file found to calibrate.')
            return

        infile = csv_for_tip
        base, _ = os.path.splitext(infile)
        tipfile = base + '.tip'

        try:
            tip, rmse = pivot_calib_lsq(infile, 2.0, tipfile)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Tip calibration failed: {e}')
            return

        idx = self.toolid.currentIndex()
        self.tipCalFiles[idx] = tipfile
        self.tipCalData[idx, :] = np.loadtxt(tipfile, delimiter=',')
        self.updateToolDefDisplay()
        self.log(f'Tip calibration saved to {tipfile} (RMSE {rmse:.3f} mm)')


    def on_connect(self):
        # --- UI lockdown ---
        for widget in (self.comport, self.tooldefbutton, self.tooldefclearbutton,
                       self.tipcalbutton, self.tipcalclearbutton,
                       self.outputfileselectbutton, self.outputfileclearbutton):
            widget.setEnabled(False)
        self.buttons['Connect'].setEnabled(False)
        #self.buttons['Register Phantom'].setEnabled(False)
        
        self.resetToolStatusIndicators()

        useEth = (self.connType.currentText() == 'Ethernet')

        # --- Validate tool‐definition files ---
        self.toolsUsed = [i+1 for i,(f,_) in enumerate(self.toolDefFiles) if f]
        if not self.toolsUsed:
            QMessageBox.warning(self, 'Warning', 
                                'No tool definition file(s) specified!')
            # self.buttons['Connect'].setEnabled(True)
            # return

        # # --- Build Polaris command strings ---   ...> **This for "Ethernet commiunication without using NDI SDK"!!!**
        # maxTool = max(self.toolsUsed)
        # if maxTool < 4:
        #     self.gx_cmd_str = 'GX:800B' if self.rbtrack.isChecked() else 'GX:9000'
        #     self.pstat_cmd_str = 'PSTAT:801f'
        # else:
        #     if not self.rbtrack.isChecked():
        #         QMessageBox.warning(self, 'Warning', 'Tool ID only supported on ports A–C.')
        #         self.buttons['Connect'].setEnabled(True)
        #         return
        #     self.gx_cmd_str = 'GX:A00B'
        #     self.pstat_cmd_str = 'PSTAT:A01f'

        # # --- Open comms ---
        # if useEth:
        #     host = self.ipAddress.text()
        #     try:
        #         port = int(self.ipPort.text())
        #     except ValueError:
        #         QMessageBox.warning(self, 'Warning', 'IP Port must be an integer!')
        #         self.buttons['Connect'].setEnabled(True)
        #         return
        #     try:
        #         self.fidSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         self.fidSocket.settimeout(0.1)
        #         self.fidSocket.connect((host, port))
        #     except Exception as e:
        #         QMessageBox.critical(self, 'Error',
        #                              f'Failed to open Ethernet to {host}:{port}: {e}')
        #         self.buttons['Connect'].setEnabled(True)
        #         return
        #     self.log(f'Connected via Ethernet to {host}:{port}')
        # else:
        #     port = self.comport.currentText()
        #     if port == 'Auto':
        #         # existing registry lookup…
        #         try:
        #             res = subprocess.check_output([
        #                 'reg', 'query', r'HKLM\HARDWARE\DEVICEMAP\SERIALCOMM'
        #             ], stderr=subprocess.DEVNULL).decode()
        #             ports = [l.split()[-1] for l in res.splitlines() if 'REG_SZ' in l]
        #             if len(ports) == 1:
        #                 port = ports[0]
        #             else:
        #                 raise RuntimeError
        #         except Exception:
        #             QMessageBox.warning(self, 'Warning',
        #                                 'Multiple or no COM ports found; set manually.')
        #             self.buttons['Connect'].setEnabled(True)
        #             return
        #     try:
        #         self.fidSerial = serial.Serial(port, baudrate=9600, timeout=0.2)
        #     except Exception as e:
        #         QMessageBox.critical(self, 'Error', f'Failed to open serial port {port}: {e}')
        #         self.buttons['Connect'].setEnabled(True)
        #         return
        #     self.log(f'Connected via Serial port {port}')

        #     # flush any stray data
        #     self.fidSerial.send_break(0.01)
        #     time.sleep(3)
        #     try:
        #         _ = self._polarisGetResponse()
        #     except Exception as e:
        #         self.log(f'Initial flush failed: {e}')

        # # --- Polaris initialization ---
        # self.baseUnixTimestamp = time.time()
        # for cmd in ['BEEP:1','COMM:40000','BEEP:2','INIT:','VSEL:1','IRATE:0']:
        #     # on serial we’ve already flushed once; on Ethernet we start fresh
        #     self._polarisSendCommand(cmd, debug=True)
        #     _ = self._polarisGetResponse()

        # # --- Upload tool definitions ---
        # for tnum in self.toolsUsed:
        #     fn, typ = self.toolDefFiles[tnum-1]
        #     with open(fn, 'rb') as f:
        #         pos = 0
        
        #         while True:
        #             chunk = f.read(64)
        #             if not chunk:
        #                 break
        #             hexdata = ''.join(f'{b:02X}' for b in chunk).ljust(128, 'F')
        #             cmd = f'PVWR:{chr(self.BASE_TOOL_CHAR+tnum)}{pos:04X}{hexdata}'
        #             self._polarisSendCommand(cmd, debug=True)
        #             _ = self._polarisGetResponse()
        #             pos += 64
        #     for sub in (f'PINIT:{chr(self.BASE_TOOL_CHAR+tnum)}',
        #                 f'PENA:{chr(self.BASE_TOOL_CHAR+tnum)}{typ}'):
        #         self._polarisSendCommand(sub, debug=True)
        #         _ = self._polarisGetResponse()

        # # --- Start data streaming ---
        # self._polarisSendCommand(self.pstat_cmd_str, debug=True)
        # _ = self._polarisGetResponse()
        # self._polarisSendCommand('TSTART:', debug=True)
        # _ = self._polarisGetResponse()

        # # --- Prepare CSV output ---
        # out = self.outputFilePath or ''
        # if not out:
        #     existing = glob.glob('trial*.csv')
        #     idxs = [int(f[5:-4]) for f in existing if f[5:-4].isdigit()]
        #     num = max(idxs)+1 if idxs else 1
        #     out = f'trial{num:03d}.csv'
        # self.outputFilePath = out
        # self.updateOutputFilePath()
        # self.fidDataOut = open(self.outputFilePath, 'w')
        # self.fidDataOut.write(
        #     'polaris_time,unix_time,tool_id,q0,q1,q2,q3,tx,ty,tz,tx_tip,ty_tip,tz_tip,reg_err,capture_note\n'
        # )

        # # --- Re‐enable UI ---
        # for btn in self.buttons.values():
        #     btn.setEnabled(True)
        
        if useEth:
            # Build NDITracker settings
            settings = {
                'tracker type': 'vega',
                'ip address':   self.ipAddress.text(),
                'port':         int(self.ipPort.text()),
                'romfiles':     [fn for fn,_ in self.toolDefFiles if fn]
            }
            try:
                # Start the Polaris Vega XT over Ethernet
                self.ndiTracker = NDITracker(settings)
                self.ndiTracker.start_tracking()
                self.log(f"NDITracker started at {settings['ip address']}:{settings['port']}")
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to start NDITracker:\n{e}')
                self.buttons['Connect'].setEnabled(True)
                return

        # Prepare CSV output
        out = self.outputfile.text().strip()          # whatever the user typed
        if out and not out.lower().endswith('.csv'):
            out += '.csv'
            
        # NEW ----------------------------------------------------------
        while out and Path(out).exists():
            base, num = re.match(r'(.*?)(\d{3})\.csv$', out).groups()
            out = f"{base}{int(num)+1:03d}.csv"
            
        if not out:                                   # box left blank
            existing = glob.glob('trial*.csv')
        # out = self.outputFilePath or ''
        # if not out:
        #     existing = glob.glob('trial*.csv')
            idxs = [int(f[5:-4]) for f in existing if f[5:-4].isdigit()]
            num = max(idxs)+1 if idxs else 1
            out = f'trial{num:03d}.csv'
            
        self.outputFilePath = out
        self.updateOutputFilePath()
        
        # self.fidDataOut = open(self.outputFilePath, 'w')
        # # self.fidDataOut.write(
        # #     'polaris_time,unix_time,tool_id,q0,q1,q2,q3,tx,ty,tz,tx_tip,ty_tip,tz_tip,reg_err,capture_note\n'
        # # ) # this would be the fromat of the ouptut CSV file
        # self.fidDataOut.write(
        #     'polaris_time,unix_time,tool_id,q0,q1,q2,q3,tx,ty,tz,tx_tip,ty_tip,tz_tip,reg_err,capture_note\n'
        #     ) #'polaris_time,unix_time,tool_id,qw,qx,qy,qz,tx,ty,tz,tx_tip,ty_tip,tz_tip,reg_err,capture_note\n'


        # Re-enable UI buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
            
        self.lastFinishedCSV = ''
        self._set_tip_button_enabled()   
        
        # Initially we are idle: no capture, so Stop/Pause must be disabled
        self.buttons['Stop'].setEnabled(False)
        self.buttons['Pause'].setEnabled(False)
        self._paused = False

    
    def on_disconnect(self):
        # self.ndi_frame0 = None
        # self.host_t0 = None
        # self.POLARIS_HZ = None
        # self._rate_est = None
        # self._rate_samples = 0
        # self._using_default_rate = True


        if getattr(self, 'ndiTracker', None):
            try:
                self.ndiTracker.stop_tracking()
            except:
                pass
            self.ndiTracker = None

        if getattr(self, 'fidDataOut', None):
            try:
                self.fidDataOut.flush()
                self.fidDataOut.close()
            finally:
                self.fidDataOut = None

        self.captureTimer.stop()
        self.resetToolStatusIndicators()

        for name, btn in self.buttons.items():
            btn.setEnabled(name == 'Connect')
        
        self.lastFinishedCSV = ''
        self._set_tip_button_enabled()
        self._paused = False
   
    
    def on_singlecap(self):
        self.ndi_frame0 = None
        self.host_t0 = None
        self.POLARIS_HZ = None
        self._rate_est = None
        self._rate_samples = 0
        self._using_default_rate = True

        self.previewFlag = False
        self.captureTimer.stop()
        self.endTrackingFlag = True
        self.buttons['Calibrate Tip'].setEnabled(False)   # disable during capture

        self._open_csv_if_needed()  # opens current outputfile 

        self.capturenote.setEnabled(False)
        for btn in ['Disconnect','Start Cap','Start Cap + Video','Preview','Pause']:
            self.buttons[btn].setEnabled(False)
            
        self.buttons['Stop'].setEnabled(False)

        status = self._captureData()
        if status != 0:
            QMessageBox.warning(self, 'Warning', 'Could not capture any data!')

        # self._polarisSendCommand('BEEP:1', debug=True) ---> not used in Ethernet mode!!

        for btn in ['Disconnect','Start Cap','Start Cap + Video','Preview']:
            self.buttons[btn].setEnabled(True)
            
        self.buttons['Stop'].setEnabled(False)
        self.buttons['Pause'].setEnabled(False)

        # finish handling depending on append mode
        if self.cbAppendSingles.isChecked():
            # Append mode: close but DO NOT roll; next Single Cap will append to same file
            if getattr(self, 'fidDataOut', None):
                try:
                    self.fidDataOut.flush()
                    self.fidDataOut.close()
                finally:
                    self.fidDataOut = None
            # expose the file to the Tip Cal button if want to use it mid-session
            self.lastFinishedCSV = self.outputFilePath
        else:
            # Original behavior (one file per Single Cap)
            if getattr(self, 'fidDataOut', None):
                try:
                    self.fidDataOut.flush()
                    self.fidDataOut.close()
                finally:
                    self.fidDataOut = None
            self.lastFinishedCSV = self.outputFilePath
            if self.outputFilePath:
                self.outputFilePath = self._next_trial_name(self.outputFilePath)
                self.updateOutputFilePath()

        self._update_capture_note_enable()
        self._set_tip_button_enabled()


    def on_startcap(self):
        # self.ndi_frame0 = None
        # self.host_t0 = None
        # self.POLARIS_HZ = None
        # self._rate_est = None
        # self._rate_samples = 0
        # self._using_default_rate = True


        # # disable buttons & note during streaming
        # self.previewFlag = False              # record mode
        # self._open_csv_if_needed()            # open & write header only now

        # self.capturenote.setEnabled(False)
        # for btn in ['Single Cap','Start Cap','Start Cap + Video','Disconnect','Preview']:
        #     self.buttons[btn].setEnabled(False)
        # self.buttons['Stop'].setEnabled(True)

        # self.endTrackingFlag = False
        # ###HyperDecks if requested by "Start Cap + Video"
        # if self.send_video_cmd:
        #     self.send_video_cmd = False  # one-shot
        #     try:
        #         self._video_start()
        #     except Exception as e:
        #         self.log(f"[HyperDeck] Start failed: {e}")
        #         ###

        # self.captureTimer.start(0)
        
        # Reset timing only for a fresh run, not for pause→resume
        if not self._paused:
            self.ndi_frame0 = None
            self.host_t0 = None
            self.POLARIS_HZ = None
            self._rate_est = None
            self._rate_samples = 0
            self._using_default_rate = True

        # disable buttons & note during streaming
        self.previewFlag = False              # record mode
        self._open_csv_if_needed()            # open & write header only now..

        self.capturenote.setEnabled(False)
        for btn in ['Single Cap','Start Cap','Start Cap + Video','Disconnect','Preview']:
            self.buttons[btn].setEnabled(False)
        self.buttons['Stop'].setEnabled(True)
        self.buttons['Pause'].setEnabled(True)   # allow pausing

        self.endTrackingFlag = False
        ### HyperDecks if requested by "Start Cap + Video"
        if self.send_video_cmd:
            self.send_video_cmd = False  # one-shot
            try:
                self._video_start()
            except Exception as e:
                self.log(f"[HyperDeck] Start failed: {e}")
        ###
        self.captureTimer.start(0)
        self._paused = False
        # tip calibration stays disabled while file is open
        self._set_tip_button_enabled()
        
    def on_pausecap(self):
            """
            Pause continuous streaming without closing the CSV.
            can edit the capture note and then hit Start Cap again
            to continue writing into the SAME file.
            """
            if self.previewFlag:
                # In preview mode we don’t have an open CSV anyway.
                self.endTrackingFlag = True
                self.captureTimer.stop()
            else:
                self.endTrackingFlag = True
                self.captureTimer.stop()

            self._paused = True

            # Allow user to change the capture note between segments
            self.capturenote.setEnabled(True)
            self._update_capture_note_enable()

            # Re-enable controls needed while paused
            self.buttons['Stop'].setEnabled(False)
            self.buttons['Pause'].setEnabled(False)
            for btn in ['Single Cap', 'Start Cap', 'Start Cap + Video', 'Disconnect', 'Preview']:
                self.buttons[btn].setEnabled(True)

            self._set_tip_button_enabled()

  
    def on_startcapvid(self):
    #   self.previewFlag = True
    #   self.on_startcap()
        self.previewFlag = False              # MUST be False so record...
        self.send_video_cmd = True            
        self.on_startcap()
        
    
    def on_stopcap(self):
        self.endTrackingFlag = True
        self.captureTimer.stop()  # <- stop the timer
        self._paused = False

        # close CSV if we were recording
        if getattr(self, 'fidDataOut', None):
            try:
                self.fidDataOut.flush()
                self.fidDataOut.close()
            finally:
                self.fidDataOut = None
        # >>> record the just-finished file before rolling ..
        self.lastFinishedCSV = self.outputFilePath

        # roll the suggested name to the next index so the next capture starts fresh
        if self.outputFilePath:
            self.outputFilePath = self._next_trial_name(self.outputFilePath)
            self.updateOutputFilePath()

        self.buttons['Stop'].setEnabled(False)
        self.buttons['Pause'].setEnabled(False)
        for btn in ['Single Cap','Start Cap','Start Cap + Video','Disconnect','Preview']:
            self.buttons[btn].setEnabled(True)

        ### Stop HyperDecks if they are recording
        if self.video_running:
            try:
                self._video_stop()
            except Exception as e:
                self.log(f"[HyperDeck] Stop failed: {e}")


        self._update_capture_note_enable()
        self._set_tip_button_enabled()
        
        
    def _next_trial_name(self, path: str) -> str:
        m = re.match(r'(.*?)(\d{3})\.csv$', path)
        if m:
            base, num = m.groups()
            return f"{base}{int(num)+1:03d}.csv"
        
        # fallback: auto-create trial###.csv
        existing = glob.glob('trial*.csv')
        nums = [int(p[5:-4]) for p in existing if re.match(r'trial\d{3}\.csv$', os.path.basename(p))]
        nxt = (max(nums) + 1) if nums else 1
        
        return f"trial{nxt:03d}.csv"

            
    def _captureIteration(self):
        """
        Called by QTimer.  Runs one capture & UI update, but
        traps any unexpected errors so the timer keeps running.
        """
        if self.endTrackingFlag:
            return

        try:
            status = self._captureData()
        except Exception as e:
            # Log it and keep going
            self.log(f"[ERROR] in _captureData: {e}")
            status = -1

        # Let Qt handle paints & events
        QApplication.processEvents()

        # if too many consecutive fails:
        # if status != 0:
        #     self.captureTimer.stop()


    def on_preview(self):
        # self.previewFlag = True
        # self.on_startcap()
        self.ndi_frame0 = None
        self.host_t0 = None
        self.POLARIS_HZ = None
        self._rate_est = None
        self._rate_samples = 0
        self._using_default_rate = True


        self.previewFlag = True               # preview mode = NO WRITES

        self.capturenote.setEnabled(False)
        for btn in ['Single Cap','Start Cap','Start Cap + Video','Disconnect','Preview']:
            self.buttons[btn].setEnabled(False)
        self.buttons['Stop'].setEnabled(True)
        self.buttons['Pause'].setEnabled(False)

        self.endTrackingFlag = False
        self.captureTimer.start(0)
        self._set_tip_button_enabled()  # will disable during recording
        

    def on_tooldef_select(self):
        idx = self.toolid.currentIndex()
        fn, _ = QFileDialog.getOpenFileName(self
                                            , 'Select ROM file',
                                            filter='ROM Files (*.rom)'
                                            )
        if fn:
            self.toolDefFiles[idx] = [fn, 'D']
            self.ptd_rom_path.setText(os.path.basename(fn))
            self.updateToolDefDisplay()
            

    def on_tooldef_clear(self):
        idx = self.toolid.currentIndex()
        self.toolDefFiles[idx] = ['', '']
        self.ptd_rom_path.clear()
        self.updateToolDefDisplay()
        

    def on_tipcal_select(self):
        idx = self.toolid.currentIndex()
        fn, _ = QFileDialog.getOpenFileName(
            self, 
            'Select TIP file', 
            filter='TIP Files (*.tip)'
            )
        
        if fn:
            data = np.loadtxt(fn, delimiter=',')
            if data.size != 3:
                QMessageBox.warning(self, 
                                    'Warning',
                                    'Invalid tip file!')
            else:
                self.tipCalFiles[idx] = fn
                self.tipCalData[idx, :] = data
                
                self.ptd_tip_path.setText(os.path.basename(fn))
                self.tipcalfile.setText(os.path.basename(fn))
                
                self.updateToolDefDisplay()
                

    def on_tipcal_clear(self):
        idx = self.toolid.currentIndex()
        self.tipCalFiles[idx] = ''
        self.tipCalData[idx, :] = 0
        
         # clear both displays
        self.ptd_tip_path.clear()
        self.tipcalfile.setText('No tip calibration file selected!')
        
        self.updateToolDefDisplay()
        

    def on_outputfile_select(self):
        fn, _ = QFileDialog.getSaveFileName(self, 
                                            'Select CSV file', 
                                            filter='CSV Files (*.csv)')
        if fn:
            self.outputFilePath = fn
            self.updateOutputFilePath()
            
    
    def _set_tip_button_enabled(self):
        """Enable 'Calibrate Tip' iff we are NOT recording and we have a finished CSV with data."""
        recording = self.fidDataOut is not None
        path = self.lastFinishedCSV  # only calibrate against the *finished* file
        ok = False
        if (not recording) and path:
            p = Path(path)
            if p.exists():
                try:
                    # Require at least header + 1 data line 
                    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read two lines max
                        has_two = sum(1 for _, __ in zip(f, range(2))) == 2
                    ok = has_two
                except Exception:
                    # Fallback to size check if needed
                    ok = p.stat().st_size > 64
                    
        self.buttons['Calibrate Tip'].setEnabled(ok)

        # Optional: show the file that will be used for calibration in the box
        if ok:
            self.outputfile.setText(os.path.basename(path))


    def on_outputfile_clear(self):
        self.outputFilePath = ''
        self.updateOutputFilePath()

    
    def on_generateROM(self):
        raw_csv = self.outputfile.text()
        if not raw_csv.endswith('.csv'):
            QMessageBox.warning(self, 
                                "Warning", 
                                "No CSV trial file to generate ROM from!")
            return

        base, _ = os.path.splitext(raw_csv)
        rom_file = base + '.rom'
        try:
            self.generate_rom_from_csv(raw_csv, rom_file)
        except Exception as e:
            QMessageBox.critical(self, 
                                 "Error",
                                 f"ROM generation failed:\n{e}")
            return
        QMessageBox.information(self, 
                                "Success",
                                f"Wrote ROM: {rom_file}")


    def generate_rom_from_csv(raw_csv: str, rom_filename: str):
      """
      1) Load the raw CSV of marker positions: each row is
       x1,y1,z1,x2,y2,z2,… for all N markers.
      2) For each frame, register via ICP to our nominal marker model.
      3) Collect all in‐lier registrations, average them to get final marker positions.
      4) Pack those into the NDI .rom binary format.
      """
      # --- 1) Load raw data ----------------------------------------------------
      all_data = np.loadtxt(raw_csv, delimiter=',', skiprows=1)
      num_frames, num_cols = all_data.shape
      num_markers = num_cols // 3
      if num_markers * 3 != num_cols:
        raise RuntimeError(f"Expected 3*M columns, got {num_cols}")

      # --- 2) Build nominal marker model ---------------------------------------
      diam = 2.3 * 25.4  # marker circle diameter in mm
      angs = np.deg2rad([41, 119, 167, 219, 319])  # the 5 nominal angles
      
      # (M×3) array of expected marker positions in local CS
      nominal = np.vstack([
        (diam/2)*np.cos(angs),
        (diam/2)*np.sin(angs),
        np.zeros_like(angs)
      ]).T

      target_pc = o3d.geometry.PointCloud()
      target_pc.points = o3d.utility.Vector3dVector(nominal)

      # --- 3) ICP per frame, collect the registered points ---------------------
      regs = []
      for row in all_data:
        pts = row.reshape(num_markers, 3)
        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(pts)

        res = o3d.registration.registration_icp(
            src_pc, target_pc,
            max_correspondence_distance=5.0,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint()
        )
        if res.inlier_rmse < 1.0:   
            # apply the estimated transform to the nominal model to get
            # where the nominal markers ended up in tracker :
            T = res.transformation  # 4×4
            homog = np.hstack([nominal, np.ones((num_markers,1))])  # M×4
            reg_pts = (T @ homog.T).T[:, :3]  # M×3
            regs.append(reg_pts)
      if not regs:
        raise RuntimeError("No valid ICP registrations found")

      # --- 4) Average over all frames -----------------------------------------
      estimate = np.mean(np.stack(regs), axis=0)  # M×3

      # --- 5) Write .rom file --------------------------------------------------
      # The following header fields :
      subtype       = 0x01
      toolType      = 0x02
      toolRev       = 0
      seqNum        = 0
      maxAngle      = 90
      numMarkers    = estimate.shape[0]
      minMarkers    = 3
      max3DError    = int(0.5 * 1000)  # [mm] 
      # face & group bytes
      numFaces      = 1
      numGroups     = 1
      faceGrpByte   = (numFaces << 3) | (numGroups & 0x07)

      # dateBytes: pack year, month, day-of-year & seqNum bits .....
      today = datetime.date.today()
      doy   = today.timetuple().tm_yday
      datevar = ( (today.year - 1900) << 15 ) \
            | ((today.month-1)    << 11 ) \
            | (doy                 << 2  ) \
            | ((seqNum >> 8) & 0x03)
      dateBytes = datevar

      with open(rom_filename, 'wb') as f:
        # 2-byte little-endian each
        f.write(struct.pack('<H', subtype))
        f.write(struct.pack('<H', toolType))
        f.write(struct.pack('<H', toolRev))
        f.write(struct.pack('<H', seqNum))
        f.write(struct.pack('<H', maxAngle))
        f.write(struct.pack('<H', numMarkers))
        f.write(struct.pack('<H', minMarkers))
          
        # pack max3DError as float?
        f.write(struct.pack('<f', max3DError/1000.0))
          
        # now write markerLocs (M × 3 floats)
        for x,y,z in estimate:
            f.write(struct.pack('<f', x))
            f.write(struct.pack('<f', y))
            f.write(struct.pack('<f', z))
            
        # write zeros for normals ( assume face normal = +Z)
        for _ in range(numMarkers):
            f.write(struct.pack('<fff', 0.0, 0.0, 1.0))
        
        # face normals (one only)
        f.write(struct.pack('<fff', 0.0, 0.0, 1.0))
        # dateBytes and faceGrpByte
        f.write(struct.pack('<I', dateBytes))   # 4 bytes
        f.write(struct.pack('<B', faceGrpByte)) # 1 byte

      print(f"Wrote {rom_filename}")

    
    def _polarisSendCommand(self, cmd: str, debug=False):
        """
        Send one ASCII command terminated with CR.
        • On Serial: send cmd+CRC16(cmd)+'\r'
        • On Ethernet: only if self.fidSocket exists; otherwise skip.
        """
        data = cmd.encode('ascii')

        # SERIAL path
        if getattr(self, 'fidSerial', None):
            crc_int   = int(polaris_crc16(cmd), 16)
            packet    = data + struct.pack('>H', crc_int) + b'\r'
            self.fidSerial.write(packet)

        # ETHERNET path (only if there is an opened a socket)
        elif getattr(self, 'fidSocket', None):
            packet = data + b'\r'
            self.fidSocket.sendall(packet)

        else:
            # No comms channel available → just log 
            self.log(f"[WARN] Cannot send '{cmd}': no serial or socket open")
            return

        # Log exactly what we sent
        self.log(f"> {cmd}")
        if debug:
            print(">>", cmd, "->", packet.hex() if 'packet' in locals() else None)


    def _polarisGetResponse(self) -> str:
        """
        Read until '\r'.  
        • Ethernet/TCP: tracker replies with plain ASCII (no CRC), so strip CR and return.  
        • Serial: responses include a 2-byte CRC; verify it before returning.
        """
        if self.connType.currentText() == 'Ethernet':
            # --- Ethernet: plain ASCII response ---
            raw = b''
            while not raw.endswith(b'\r'):
                chunk = self.fidSocket.recv(1)
                if not chunk:
                    raise RuntimeError('Connection closed by Polaris')
                raw += chunk
            resp = raw.rstrip(b'\r').decode('ascii', errors='ignore')
            self.log(f"< {resp}")
            return resp

        # --- Serial: CRC-checked response ---
        raw = self.fidSerial.read_until(b'\r')
        raw = raw.rstrip(b'\r')
        if len(raw) < 3:
            raise RuntimeError('Response too short to contain CRC')

        payload_bytes = raw[:-2]
        recv_crc      = raw[-2:]
        resp          = payload_bytes.decode('ascii', errors='ignore')

        expected_crc = struct.pack('>H', int(polaris_crc16(resp), 16))
        if recv_crc != expected_crc:
            got = recv_crc.hex().upper()
            exp = expected_crc.hex().upper()
            raise RuntimeError(f"CRC mismatch: got {got}, expected {exp}")

        self.log(f"< {resp}")
        return resp
    
    def _y_limit_top(self, z_abs):
        f = self.FOV
        z = float(z_abs)
        if z <= f['TOP_BREAK_Z']:
            return f['W_NEAR'] + (z - f['NEAR_Z']) * np.tan(f['TOP_ANG_STEEP'])
        y_break = f['W_NEAR'] + (f['TOP_BREAK_Z'] - f['NEAR_Z']) * np.tan(f['TOP_ANG_STEEP'])
        
        return y_break + (z - f['TOP_BREAK_Z']) * np.tan(f['TOP_ANG_SHALLOW'])

    def _x_limit_side(self, z_abs):
        f = self.FOV
        # linear ramp from near (H_NEAR) to far (H_FAR)
        t = np.clip((z_abs - f['NEAR_Z']) / (f['FAR_Z'] - f['NEAR_Z']), 0.0, 1.0)
        
        return f['H_NEAR'] + t * (f['H_FAR'] - f['H_NEAR'])

    
    # def _compute_fov_status(self, tip_xyz: np.ndarray) -> int:  %---> old
    #     """
    #     Returns 0 if tip_xyz is fully inside the Vega-XT pyramid FOV,
    #              1 if it lies exactly on a side plane,
    #              2 if it's outside.

    #     Uses the NDI Vega-XT specs:
    #       • Near plane @ 950 mm
    #       • Far  plane @ 950+2400=3350 mm
    #       • Half-width/height @ near = 1566/2 × 1312/2
    #       • Half-width/height @ far  = 1856/2 × 1470/2
    #     """
    #     # make Z positive
    #     d = abs(tip_xyz[2])
    #     # near, far = 950.0, 950.0 + 2400.0
    #     near, far = 950.0, 950.0 + 3000.0                 # mm
    #     w_near, h_near = 480/2, 448/2
    #     w_far,  h_far  = 1856/2, 1470/2
        
    #     tol = 10

    #     # 1) Z‐range
    #     if d < near - tol or d > far + tol:
    #         return 2
    #     if abs(d - near) <= tol or abs(d - far) <= tol:
    #         return 1

    #     # 2) linear interpolate the pyramid’s half‐width/height
    #     # w_near, h_near = 1566/2, 1312/2
    #     # w_far,  h_far  = 1856/2, 1470/2
        
    #     t = (d - near) / (far - near)
    #     w = w_near + (w_far - w_near) * t
    #     h = h_near + (h_far - h_near) * t

    #     x, y = tip_xyz[0], tip_xyz[1]
    #     if abs(x) > w + tol or abs(y) > h + tol:
    #         return 2
    #     if abs(abs(x) - w) <= tol or abs(abs(y) - h) <= tol:
    #         return 1
    #     return 0
    
    def _compute_fov_status(self, tip_xyz: np.ndarray) -> int:  #-->new verson
        if tip_xyz is None or np.any(np.isnan(tip_xyz)):
            return 4  # Not Loaded
        f = self.FOV
        x, y, z = float(tip_xyz[0]), float(tip_xyz[1]), abs(float(tip_xyz[2]))
        tol = f['BORDER_TOL']

        # Z bounds
        if z < f['NEAR_Z'] - tol or z > f['FAR_Z'] + tol:
            return 2
        z_on = (abs(z-f['NEAR_Z']) <= tol) or (abs(z-f['FAR_Z']) <= tol)

        # Lateral limits 
        def y_limit(z_abs):
            if z_abs <= f['TOP_BREAK_Z']:
                return f['W_NEAR'] + (z_abs - f['NEAR_Z']) * np.tan(f['TOP_ANG_STEEP'])
            yb = f['W_NEAR'] + (f['TOP_BREAK_Z']-f['NEAR_Z']) * np.tan(f['TOP_ANG_STEEP'])
            return yb + (z_abs - f['TOP_BREAK_Z']) * np.tan(f['TOP_ANG_SHALLOW'])

        def x_limit(z_abs):
            t = np.clip((z_abs - f['NEAR_Z']) / (f['FAR_Z'] - f['NEAR_Z']), 0.0, 1.0)
            return f['H_NEAR'] + t * (f['H_FAR_SIDE'] - f['H_NEAR'])

        y_lim, x_lim = y_limit(z), x_limit(z)
        if abs(y) > y_lim + tol or abs(x) > x_lim + tol:
            return 2
        on = z_on or (abs(abs(y)-y_lim) <= tol) or (abs(abs(x)-x_lim) <= tol)
        return 1 if on else 0


    # def _captureData(self) -> int:  ## ----> this is for the case without using NDI SDK
    #     """
    #     Send GX command, read back the frame, and parse tool transforms.
    #     Skips any ERROR frames (e.g. 'ERROR046') so we don’t try to int() them.
    #     """
    #     status = -1
    #     for _ in range(10):
    #         # ask Polaris for data
    #         self._polarisSendCommand(self.gx_cmd_str)
    #         resp = self._polarisGetResponse()
    #         parts = resp.split('\n')
    #         frame = parts[-1].strip()

    #         # if the tracker says "ERRORxxx", skip it
    #         if frame.startswith('ERROR') or not re.fullmatch(r'[0-9A-Fa-f]{8}', frame):
    #             self.log(f"[WARN] invalid frame—no tool: '{frame}'")
    #             time.sleep(0.01)
    #             continue

    #         # now we have exactly 8 hex digits per tool
    #         note = self.capturenote.text().replace(',', '')
    #         for idx, tnum in enumerate(self.toolsUsed):
    #             hexstr = frame[(tnum-1)*8 : (tnum-1)*8+8]
    #             # guaranteed now to be 8-char hex
    #             t = int(hexstr, 16) / 60.0
    #             unixt = self.baseUnixTimestamp + t

    #             dl = self.gx_transform_map[tnum-1]
    #             line = parts[dl-1]
    #             if len(line) == 51:
    #                 # decode quaternion + translation
    #                 q = np.array([int(line[i*6:(i+1)*6]) / 10000 for i in range(4)])
    #                 trans = np.array([int(line[24+i*7:24+i*7+7]) / 100 for i in range(3)])
    #                 tipg = trans + quat_rot(q, self.tipCalData[tnum-1])
    #                 err = int(line[45:51], 16) / 10000

    #                 if not self.previewFlag:
    #                     self.fidDataOut.write(
    #                         f"{t:.4f},{unixt:.2f},{chr(self.BASE_TOOL_CHAR+tnum)},"
    #                         + ",".join(f"{x:+.4f}" for x in np.hstack((q, trans, tipg)))
    #                         + f",{err:.4f},{note}\n"
    #                     )

    #                 # update 2D tracking plots
    #                 self.pointHandles[idx]['front'].set_position((trans[1], -trans[0]))
    #                 self.pointHandles[idx]['top'].set_position((-trans[2], -trans[1]))
    #                 self.pointHandles[idx]['side'].set_position((-trans[2], -trans[0]))
                    
    #                 self.canvas_front.draw()
    #                 self.canvas_top.draw()
    #                 self.canvas_side.draw()


    #                 # status‐code LED
    #                 code = parts[dl-1 + ((4 - ((dl-1)%4)))][8 - 2*((tnum-1)%3)]
    #                 sc = {'7':2, 'B':1, '3':0}.get(code, 0)
    #                 self.setToolStatusIndicator(tnum, sc)

    #                 # update CT 3D view
    #                 if self.registration_matrix is not None:
    #                     pt4 = np.hstack((tipg, 1.0))
    #                     x, y, z, _ = (self.registration_matrix @ pt4)
    #                     if self.tool_ct_handle:
    #                         self.tool_ct_handle._offsets3d = ([x], [y], [z])
    #                     else:
    #                         self.tool_ct_handle = self.ax_ct.scatter([x], [y], [z],
    #                                                                  marker='^', s=100)
    #                     self.canvas_ct.draw()

    #                 status = 0
    #             else:
    #                 # malformed line, clear that tool’s text
    #                 for handle in self.pointHandles[idx].values():
    #                     handle.set_position((np.nan, np.nan))

    #         break  # exit retry loop since we got a valid frame

    #     return status
    
    def _captureData(self) -> int:
        """
        Grab one frame from NDITracker, apply tip-offset,
        write the CSV, update 2D letters—and color them by FOV status.
        """
        try:
            port_handles, timestamps, frame_numbers, tracking, quality = \
                self.ndiTracker.get_frame()
                
            #--------------------------  for polaris_time ------------------------------
            # use only frames with valid transforms
            valid_fns = [int(fn) for fn, T in zip(frame_numbers, tracking) if not np.isnan(T).any()]

            if self.ndi_frame0 is None:
                if valid_fns:
                    self.ndi_frame0 = (min(valid_fns) & 0xFFFFFFFF)
                    self.host_t0    = time.time()
                    # start fresh
                    self._rate_est = None
                    self._rate_samples = 0
                    self._using_default_rate = True
            else:
                if valid_fns:
                    fr_now = (max(valid_fns) & 0xFFFFFFFF)
                    df     = (fr_now - self.ndi_frame0) & 0xFFFFFFFF
                    dt     = time.time() - self.host_t0
                    if df > 0 and dt > 0:
                        est = df / dt  # instantaneous Hz
                        self._rate_est = est if self._rate_est is None else (0.9*self._rate_est + 0.1*est)
                        self._rate_samples += 1
                        # lock the rate after a few samples when it looks plausible
                        if self._using_default_rate and self._rate_samples >= 5 and 40.0 <= self._rate_est <= 450.0:
                            self.POLARIS_HZ = self._rate_est
                            self._using_default_rate = False
                            self.log(f"Polaris rate locked: {self.POLARIS_HZ:.1f} Hz")
            #--------------------------------------------------------------------------------------------------------------
                    
        except Exception as e:
            self.log(f"[ERROR] Tracker read failed: {e}")
            return -1
        
        # if self.previewFlag:
        #     return 0            #  nothing written

        note = self.capturenote.text().replace(',', '')
        
        for idx, tnum in enumerate(self.toolsUsed):
            T = tracking[idx]
            ql = quality[idx]
            if np.isnan(T).any() or (ql is not None and ql < 0.01):
                continue

            # rotation & tip compute
            rot = Rotation.from_matrix(T[:3, :3])
            trans = T[:3, 3]
            qx, qy, qz, qw = rot.as_quat()
            q0, q1, q2, q3 = qw, qx, qy, qz # added, q3=qz...

            
            tip_xyz = trans + quat_rot([q0, q1, q2, q3], self.tipCalData[idx])
            
             # real registration error from tracker quality
            err_reg = float(ql) if ql is not None else 0.0
            
            # 1) write CSV
            if not self.previewFlag:
                if self.fidDataOut is None:
                    self._open_csv_if_needed()
                    
                # self.fidDataOut.write(
                #     f"{timestamps[idx]:.4f},{time.time():.2f},"
                #     f"{chr(self.BASE_TOOL_CHAR+tnum)},"
                #     f"{q0:+.4f},{q1:+.4f},{q2:+.4f},{q3:+.4f},"
                #     f"{trans[0]:+.4f},{trans[1]:+.4f},{trans[2]:+.4f},"
                #     f"{tip_xyz[0]:+.4f},{tip_xyz[1]:+.4f},{tip_xyz[2]:+.4f},"
                #     f"{err_reg:+.4f},{note}\n"
                # )
                
                fr = frame_numbers[idx]
                if self.ndi_frame0 is None or fr is None or np.isnan(fr):
                    continue  # still waiting for first valid frame baseline

                rate_for_time = self.POLARIS_HZ if self.POLARIS_HZ else self.DEFAULT_POLARIS_HZ

                fr_i  = int(fr) & 0xFFFFFFFF
                delta = (fr_i - self.ndi_frame0) & 0xFFFFFFFF
                polaris_time_s = delta / rate_for_time

                self.fidDataOut.write(
                    f"{polaris_time_s:.4f},{time.time():.2f},"
                    f"{chr(self.BASE_TOOL_CHAR+tnum)},"
                    f"{q0:+.4f},{q1:+.4f},{q2:+.4f},{q3:+.4f},"
                    f"{trans[0]:+.4f},{trans[1]:+.4f},{trans[2]:+.4f},"
                    f"{tip_xyz[0]:+.4f},{tip_xyz[1]:+.4f},{tip_xyz[2]:+.4f},"
                    f"{err_reg:+.4f},{note}\n"
                )


            # 2) compute FOV status
            status = self._compute_fov_status(tip_xyz)
            color_map = {0: 'green', 1: 'yellow', 2: 'red'}
            col = color_map[status]

            # # 3) update 2D letter color & position
            # for plane in ('front','top','side'):
            #     h = self.pointHandles[idx][plane]
            #     # same positioning 
            #     if plane == 'front':
            #         pos = (trans[1], -trans[0])
            #     elif plane == 'top':
            #         pos = (-trans[2], -trans[1])
            #     else:
            #         pos = (-trans[2], -trans[0])
            #     h.set_position(pos)
            #     h.set_color(col)

            # # 4) update LED grid
            # self.setToolStatusIndicator(tnum, status)   #uncomments newlly--- June 2025 
            
            handles = self.pointHandles[tnum - 1] # ----> newly added.....
            for plane in ('front','top','side'):
                h = handles[plane]
                if plane == 'front':
                    pos = (trans[1], -trans[0])
                elif plane == 'top':
                    pos = (-trans[2], -trans[1])
                else:
                    pos = (-trans[2], -trans[0])
                h.set_position(pos)
                h.set_color(col)

            self.setToolStatusIndicator(tnum, status)
       
        # redraw
        self.canvas_front.draw()
        self.canvas_top.draw()
        self.canvas_side.draw()
        
        #ipdb.set_trace()

        # CT 3D update 
        if self.registration_matrix is not None:
            pts = []
            
            for idx, tnum in enumerate(self.toolsUsed):
                T = tracking[idx]
                trans = T[:3,3]
                tip_xyz = trans + quat_rot(
                    Rotation.from_matrix(T[:3,:3]).as_quat().tolist(),
                    self.tipCalData[idx]
                )
                x,y,z,_ = self.registration_matrix @ np.hstack((tip_xyz,1.0))
                pts.append((x,y,z))
            xs, ys, zs = zip(*pts)
            
            if self.tool_ct_handle:
                self.tool_ct_handle._offsets3d = (list(xs), list(ys), list(zs))
            else:
                self.tool_ct_handle = self.ax_ct.scatter(xs, ys, zs, marker='^', s=100)
            self.canvas_ct.draw()

        return 0

##-------------------------------------------------------------------------------------------------##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Polaris GUI App — pass in tool ROM/TIP pairs"
    )
    parser.add_argument(
        '--tool', action='append', nargs=2, metavar=('ROM','TIP'),
        help="One ROM and its TIP file (e.g. --tool path/to/A.rom path/to/A.tip)"
    )
    args = parser.parse_args()

    # build list of (rom, tip) tuples
    tool_defs = args.tool or []

    app = QApplication(sys.argv)
    gui = PolarisGUI(tool_defs=tool_defs)
    
    gui.show()
    sys.exit(app.exec_())
    
#########################################################EN2025################################################################
###############################################################################################################################
