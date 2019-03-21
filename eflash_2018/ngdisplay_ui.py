import argparse
import json
import neuroglancer
import numpy as np
from nuggt.utils.ngutils import *
from nuggt.point_annotator import PointAnnotator
import os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",
                        type=int,
                        help="HTTP port for neurglancer server",
                        default=0)
    parser.add_argument("--bind-address",
                        help="The IP address to bind to as a webserver. "
                        "The default is 127.0.0.1 which is constrained to "
                        "the local machine.",
                        default="127.0.0.1")
    parser.add_argument("--static-content-source",
                        default=None,
                        help="The URL of the static content source, e.g. "
                        "http://localhost:8080 if being served via npm.")
    return parser.parse_args()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, viewer):
        QtWidgets.QMainWindow.__init__(self)
        self.viewer = viewer
        self.point_annotator = PointAnnotator(viewer)
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        self.z0 = None
        self.z1 = None
        self.points_file = None
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Neuroglancer display")
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget(self)
        l = QtWidgets.QVBoxLayout(self.main_widget)

        #
        # Points file UI
        #
        hls = []
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        hl.addWidget(QtWidgets.QLabel(text="Points file:"))
        self.points_file_widget = QtWidgets.QLineEdit()
        self.points_file_widget.setText("//media/share2/...")
        hl.addWidget(self.points_file_widget)
        self.points_file_browse_button = QtWidgets.QPushButton(text="Browse...")
        self.points_file_browse_button.clicked.connect(
            self.on_points_file_browse)
        hl.addWidget(self.points_file_browse_button)
        self.points_file_save_button = QtWidgets.QPushButton(text="Save...")
        self.points_file_save_button.clicked.connect(
            self.on_points_file_save
        )
        hl.addWidget(self.points_file_save_button)
        #
        # X0, X1, Y0, Y1, Z0, Z1
        #
        self.coord_widgets = {}
        for xyz in "xyz":
            hl = QtWidgets.QHBoxLayout()
            hls.append(hl)
            for _01 in (0, 1):
                label = "%s%d" % (xyz, _01)
                hl.addWidget(QtWidgets.QLabel(text=label))
                self.coord_widgets[label] = QtWidgets.QLineEdit()
                self.coord_widgets[label].setText("0")
                hl.addWidget(self.coord_widgets[label])
        #
        # Neuroglancer data source
        #
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        hl.addWidget(QtWidgets.QLabel(text="Neuroglancer source:"))
        self.neuroglancer_source_widget = QtWidgets.QLineEdit()
        self.neuroglancer_source_widget.setText(
            "precomputed://http://leviathan-chunglab.mit.edu/precomputed/???")
        hl.addWidget(self.neuroglancer_source_widget)
        #
        # Shader config for first source
        #
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        hl.addWidget(QtWidgets.QLabel(text="Intensity: "))
        self.intensity_widget = QtWidgets.QLineEdit("40.0")
        hl.addWidget(self.intensity_widget)
        self.shader_widget = QtWidgets.QComboBox()
        hl.addWidget(self.shader_widget)
        self.shader_widget.addItems(["gray",
                                     "red",
                                     "green",
                                     "blue",
                                     "cubehelix"])
        self.shader_widget.setCurrentIndex(4)
        #
        # Second Neuroglancer source
        #
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        self.use_neuroglancer_second_source_widget = QtWidgets.QCheckBox(
            "Neuroglancer source #2")
        hl.addWidget(self.use_neuroglancer_second_source_widget)
        self.neuroglancer_second_source_widget = QtWidgets.QLineEdit()
        self.neuroglancer_second_source_widget.setText(
            "precomputed://http://leviathan-chunglab.mit.edu/precomputed/???")
        hl.addWidget(self.neuroglancer_second_source_widget)
        #
        # Shader for second source
        #
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        hl.addWidget(QtWidgets.QLabel(text="Intensity: "))
        self.second_intensity_widget = QtWidgets.QLineEdit("40.0")
        hl.addWidget(self.second_intensity_widget)
        self.second_shader_widget = QtWidgets.QComboBox()
        hl.addWidget(self.second_shader_widget)
        self.second_shader_widget.addItems([
            "gray",
            "red",
            "green",
            "blue",
            "cubehelix"])
        self.second_shader_widget.setCurrentIndex(2)

        #
        # Display button
        #
        hl = QtWidgets.QHBoxLayout()
        hls.append(hl)
        self.display_button_widget = QtWidgets.QPushButton(
            text="Update display")
        self.display_button_widget.clicked.connect(
            self.on_update_display)
        hl.addWidget(self.display_button_widget)
        for hl in hls:
            l.addLayout(hl)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def on_points_file_browse(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open points file",
        filter="Coordinates (*.json);;All files (*)")
        self.points_file_widget.setText(filename)

    def on_points_file_save(self):
        if self.x0 is not None:
            self.replace_points()
        filename = self.points_file_widget.text()

        if os.path.exists(filename):
            msg = QtWidgets.QMessageBox.question(
                self, "Overwriting file",
                "Are you sure you want to overwrite %s?" %
                os.path.split(filename)[1])
            if msg != QtWidgets.QMessageBox.Yes:
                return
        with open(filename, "w") as fd:
            json.dump(self.all_points.tolist(), fd)
        self.points_file = filename

    def on_update_display(self):
        points_file = self.points_file_widget.text()
        if points_file == self.points_file:
            replace = True
        else:
            try:
                with open(points_file, "r") as fd:
                    self.all_points = np.array(json.load(fd))
                self.points_file = points_file
                replace = False
            except IOError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "I/O Error",
                    "Could not open %s" % points_file)
                return
            except json.JSONDecodeError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "I/O Error",
                    "%s could not be read as a JSON file" % points_file)
                return
            except UnicodeDecodeError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "I/O Error",
                    "%s could not be read as a JSON file" % points_file)
                return

        try:
            x0 = int(self.coord_widgets["x0"].text())
            x1 = int(self.coord_widgets["x1"].text())
            y0 = int(self.coord_widgets["y0"].text())
            y1 = int(self.coord_widgets["y1"].text())
            z0 = int(self.coord_widgets["z0"].text())
            z1 = int(self.coord_widgets["z1"].text())
        except ValueError:
            QtWidgets.QErrorMessage("Coordinates must be integers")
            return
        self.update_points(x0, x1, y0, y1, z0, z1, replace=replace)
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.z0, self.z1 = z0, z1
        shader_txt = self.shader_widget.currentText()
        if shader_txt == "gray":
            shader = gray_shader
        elif shader_txt == "red":
            shader = red_shader
        elif shader_txt == "green":
            shader = green_shader
        elif shader_txt == "blue":
            shader = blue_shader
        else:
            shader = cubehelix_shader
        try:
            intensity = float(self.intensity_widget.text())
        except ValueError:
            QtWidgets.QErrorMessage("Intensity must be a number")
            return
        with self.viewer.txn() as txn:
            txn.layers["image"] = neuroglancer.ImageLayer(
                source=self.neuroglancer_source_widget.text(),
                shader= shader % intensity
            )
            if self.use_neuroglancer_second_source_widget.isChecked():
                shader_txt = self.second_shader_widget.currentText()
                if shader_txt == "gray":
                    shader = gray_shader
                elif shader_txt == "red":
                    shader = red_shader
                elif shader_txt == "green":
                    shader = green_shader
                elif shader_txt == "blue":
                    shader = blue_shader
                else:
                    shader = cubehelix_shader
                try:
                    intensity = float(self.second_intensity_widget.text())
                except ValueError:
                    QtWidgets.QErrorMessage("Intensity must be a number")
                    return
                txn.layers["image-2"] = neuroglancer.ImageLayer(
                    source=self.neuroglancer_second_source_widget.text(),
                    shader = shader % intensity
                )
            else:
                for i, layer in enumerate(txn.layers):
                    if layer.name == "image-2":
                        del txn.layers[i]
                        break

    def update_points(self, x0, x1, y0, y1, z0, z1, replace):
        if replace:
            self.replace_points()

        idxs = np.where(
            (self.all_points[:, 0] >= x0) & (self.all_points[:, 0] < x1) &
            (self.all_points[:, 1] >= y0) & (self.all_points[:, 1] < y1) &
            (self.all_points[:, 2] >= z0) & (self.all_points[:, 2] < z1))[0]
        if len(idxs) == 0:
            self.point_annotator.set_points(np.zeros((0, 3)))
        else:
            self.point_annotator.set_points(self.all_points[idxs])

    def replace_points(self):
        idxs = np.where(
            (self.all_points[:, 0] >= self.x0) &
            (self.all_points[:, 0] < self.x1) &
            (self.all_points[:, 1] >= self.y0) &
            (self.all_points[:, 1] < self.y1) &
            (self.all_points[:, 2] >= self.z0) &
            (self.all_points[:, 2] < self.z1))[0]
        self.all_points = np.vstack((
            np.delete(self.all_points, idxs, 0),
            self.point_annotator.all_points
        ))


def main():
    app = QtWidgets.QApplication(sys.argv)
    args = parse_args()
    if args.static_content_source is not None:
        neuroglancer.set_static_content_source(
            url=args.static_content_source)
    neuroglancer.set_server_bind_address(
    args.bind_address, bind_port=args.port)
    viewer = neuroglancer.Viewer()
    print("Neuroglancer URL: %s" % str(viewer))
    window = ApplicationWindow(viewer)
    window.show()
    sys.exit(app.exec())


if __name__=="__main__":
    main()