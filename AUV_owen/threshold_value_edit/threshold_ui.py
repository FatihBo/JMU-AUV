# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widget.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(942, 247)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Widget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(Widget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.horizontalSlider = QtWidgets.QSlider(Widget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider)
        self.label_2 = QtWidgets.QLabel(Widget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.horizontalSlider_2 = QtWidgets.QSlider(Widget)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_2)
        self.label_3 = QtWidgets.QLabel(Widget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.horizontalSlider_3 = QtWidgets.QSlider(Widget)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_3)
        self.label_4 = QtWidgets.QLabel(Widget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.horizontalSlider_4 = QtWidgets.QSlider(Widget)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_4)
        self.label_5 = QtWidgets.QLabel(Widget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.horizontalSlider_5 = QtWidgets.QSlider(Widget)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_5)
        self.label_6 = QtWidgets.QLabel(Widget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.horizontalSlider_6 = QtWidgets.QSlider(Widget)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_6)
        self.lineEdit = QtWidgets.QLineEdit(Widget)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.SpanningRole, self.lineEdit)
        self.horizontalLayout.addLayout(self.formLayout)

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Widget"))
        self.label_7.setText(_translate("Widget", "TextLabel"))
        self.label.setText(_translate("Widget", "H最大值"))
        self.label_2.setText(_translate("Widget", "H最小值"))
        self.label_3.setText(_translate("Widget", "S最大值"))
        self.label_4.setText(_translate("Widget", "S最小值"))
        self.label_5.setText(_translate("Widget", "V最大值"))
        self.label_6.setText(_translate("Widget", "V最小值"))

