language: PYTHON
name:     "face"

variable {
 name: "output0"
 type: INT
 size: 1
 min:  5
 max:  30
}

variable {
 name: "output1"
 type: INT
 size: 1
 min:  10
 max:  100
}

variable {
 name: "output2"
 type: INT
 size: 1
 min:  10
 max:  200
}

variable {
 name: "btlneck"
 type: INT
 size: 1
 min:  50
 max:  300
}

variable {
 name: "loss_weights"
 type: FLOAT
 size: 3
 min:  0
 max:  1
}

variable {
 name: "stepsize"
 type: INT
 size: 1
 min:  10000
 max:  120000
}

variable {
 name: "base_lr"
 type: FLOAT
 size: 1
 min:  0.005
 max:  0.01
}

