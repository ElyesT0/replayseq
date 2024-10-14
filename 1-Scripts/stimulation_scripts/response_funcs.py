from expyriment import design, control, stimuli, io, misc
from expyriment.misc._timer import get_time

trigger_to_forp = {
'port1_4':'LR',
'port1_8':'LG',
'port1_2':'LY',
'port2_2':'RR',
'port2_4':'RG',
'port2_8':'RY',
'port1_1':'RB',
'port3_8':'LB',
}


"""
How to read ? -- Those are the response buttons.
There are two button box : Right (R) and Left (L).
On each button box, there are 4 buttons: Red (R), Green (G), Yellow (Y), Blue (B)

The codes correspond to Box-Button. Example:
LR: Left box, Red Button
LY: Left box, Yellow Button
RR: Right box, Red Button 
...

NOTES: Checking ports on 27-06-2024
-- port1_1 : RB,
-- port2_8 : RY,
-- port2_4 : RG,
-- port2_2 : RR,
-- port3_8 : LB,
-- port1_2: LY,
-- port1_8: LG,
-- port1_4: LR.
"""

class response_in_MEG(object):
    port1 = []
    port2 = []
    port3 = []

    def __init__(self, exp, port1Num, port2Num, port3Num):
        # from psychopy import parallel

        self.exp = exp
        # only works at the MEG. WORKS ONLY IF THE SUBJECT PRESS THE RED BUTTONS ON BOTH RESPON PANELS
        self.port1 = io.ParallelPort(port1Num)
        self.port2 = io.ParallelPort(port2Num)
        self.port3 = io.ParallelPort(port3Num)
        _ = self.port1.read_status()
        _ = self.port2.read_status()
        _ = self.port3.read_status()

        self.port1_baseline_value = self.port1.read_status()
        self.port2_baseline_value = self.port2.read_status()
        self.port3_baseline_value = self.port3.read_status()
        self.port1_last_value = self.port1_baseline_value
        self.port2_last_value = self.port2_baseline_value
        self.port3_last_value = self.port3_baseline_value


    #----------------------------------------
    # Check if subject responded.
    # Return 0 if not; 1 or 2 if they did; and -1 if they clicked ESC
    def checkResponse(self):
        # if userPressedEscape():
        #     return -1
        #-- Check if exactly one button was pressed

        # Here we apply some small tricky correction for port whose return is always non-null
        # TODO check for consistency.
        resp1 = self.port1.read_status() - self.port1_baseline_value
        resp2 = self.port2.read_status() - self.port2_baseline_value
        resp3 = self.port3.read_status() - self.port3_baseline_value

        if (resp1 != 0 and resp2 == 0 and resp1 != self.port1_last_value):# and resp3 == 0):
            self.port1_last_value = resp1
            print(f'FROM Response_funcs module::: port1_{resp1 + self.port1_baseline_value}')
            return f'port1_{resp1 + self.port1_baseline_value}'
        if (resp1 == 0 and resp2 != 0 and resp2 != self.port2_last_value):# and resp3 == 0):
            self.port2_last_value = resp2
            print(f'FROM Response_funcs module::: port2_{resp2 + self.port2_baseline_value}')
            return f'port2_{resp2 + self.port2_baseline_value}'
        if (resp1 == 0 and resp2 == 0 and resp3 != 0 and resp3 != self.port3_last_value):
            self.port3_last_value = resp3
            print(f'FROM Response_funcs module::: port3_{resp3 + self.port3_baseline_value}')
            return f'port3_{resp3 + self.port3_baseline_value}'

        if (resp1 != self.port1_last_value):
            self.port1_last_value = resp1
        if(resp2 != self.port2_last_value):
            self.port2_last_value = resp2
        if(resp3 != self.port3_last_value):
            self.port3_last_value = resp3

        return None



    def wait(self,  codes=None, duration=None, no_clear_buffer=False):

        """Homemade wait for MEG response buttons

        Parameters
        ----------
        codes : int or list, optional !!! IS IGNORED AND KEPT ONLY FOR CONSISTENCY WITH THE KEYBOARD METHOD
            bit pattern to wait for
            if codes is not set (None) the function returns for any
            event that differs from the baseline
        duration : int, optional
            maximal time to wait in ms
        no_clear_buffer : bool, optional
            do not clear the buffer (default = False)
        """
        start = get_time()
        rt = None
        if not no_clear_buffer:
            _ = self.port1.read_status()
            _ = self.port2.read_status()
            _ = self.port3.read_status()
            self.exp.keyboard.clear()
        while True:

            found = self.checkResponse()
            if found :
                rt = int((get_time() - start) * 1000)
                break

            if duration is not None:
                if int((get_time() - start) * 1000) > duration:
                    return None, None

        return found, rt



