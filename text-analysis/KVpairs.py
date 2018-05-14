import re
from text_container import TextContainer

class KVpairs(object):

    def __init__(self, text):
        self.text = self.clean_text(text)
        self.stage_num = self._find_stage_num()
        self.stage_top, self.stage_bottom = self._find_perf()
        self.avg_pump_rate = self._find_avg_pump_rate()
        self.max_pump_rate = self._find_max_pump_rate()
        self.avg_pressure = self._find_avg_pressure()
        self.max_pressure = self._find_max_pressure()
        self.ISIP = self._find_ISIP()
        #TODO deal with numbers in feet somehow
        # self.num_offsets = self.get_nums()
        # self.has_num = False
        # if self.num_offsets:
        #     self.has_num = True
        #self.feet = self.is_feet()

    def clean_text(self, text):
        new = re.sub(",", "", text)
        return new

    def get_entities(self):
        all_vars = [self.stage_num, self.stage_top, self.stage_bottom, self.avg_pump_rate, self.avg_pressure,
                    self.max_pressure, self.ISIP]
        return {k: v for d in all_vars if d for k,v in d.items()}


    def _find_stage_num(self):
        """
        Searches a block of text for the stage number
        """
        stg_num = re.compile(r"(\bstage\b|\bstg\b)\s?#?\s?(\d{1,2})", re.IGNORECASE)
        return self._get_value(stg_num, "stage number")

    def _find_perf(self):
        """
    Searches a block of text for top and bottom perforation key and values
    inputs
        text: string
    output
        TODO a list or dict or something
    TODO: test on multiple pages
          record preservation: connect to other k,v pairs of the same record
          convert to fuzzywuzzy?
    """
        perf = re.compile(
            r"\b(PERF|Perf|Perf'd|Perforated)(?:\W+\w+){0,10}?\W+(\d{2}\d{3})\b\'((?:\W+\w+){0,10}?\W+(\d{2}\d{3})\b\'){0,10}",
            re.IGNORECASE)
        matches = re.findall(perf, self.text)

        if matches:
            for match in matches:
                values = []
                for item in match:
                    if re.match(r"\d{2},\d{3}", item):
                        values.append(item)

                if values:
                    stage_top = values[0]
                    stage_bottom = values[-1]
                    return {"Stage Top perf": stage_top}, {"Stage Bottom perf": stage_bottom}
        return None, None


    def _find_avg_pump_rate(self):
        """

        """
        atr = re.compile(r"(((Avg|Average (Pump|Treating))\sRate)|\bATR\b).{1,3}(\d{1,3}\.?\d{1,2})(\sBPM)?", re.IGNORECASE)
        return self._get_value(atr, "pump rate avg")

    def _find_max_pump_rate(self):
        mpr = re.compile(r"(Max\s((treating\s|pump\s)?rate)|\bM[TP]?R\b|MAX\.? RATE).{1,3}(\d{1,3}\.?\d{1,2})(\sBPM)?", re.IGNORECASE)
        return self._get_value(mpr, "pump rate max")

    def _find_avg_pressure(self):
        """

        """
        atp = re.compile(r"(Average\s((treating\s)?pressure|Psi)|\bAT?P\b|Avg\.?\s(PRESS|Psi))\s?.{1,4}\s?(\d{1,6})(\sPSI)?", re.IGNORECASE)
        return self._get_value(atp, "average treating pressure")

    def _find_max_pressure(self):
        """

        """
        mtp= re.compile(r"(Max\s((treating\s)?pressure|Psi)|\bMT?P\b|MAX\. PRESS).{1,3}(\d{1,3}\d{1,3})(\sPSI)?", re.IGNORECASE)
        return self._get_value(mtp, "max treating pressure")

    def _find_ISIP(self):
        """
        :param self:
        :return:
        """
        isip = re.compile("(ISIP).{1,3}(\d{1,2}\d{3})", re.IGNORECASE)
        return self._get_value(isip, "ISIP")

    def _get_value(self, pattern, ent_name):
        """
        TODO deal with multiple matches
        :param pattern:
        :param message:
        :return:
        """
        matches = re.findall(pattern, self.text)
        if matches:
            start =0
            matches2 = re.finditer(pattern, self.text)
            for m in matches2:
                start = m.start()
                break
            words = self.text[start:].split()
            for w in words:
                if any(char.isdigit() for char in w):
                    num = "".join([c for c in w if c.isdigit() or c == "."])
                    return {ent_name: num}


    def get_nums(self):
        nums = []
        matches = re.findall("\d{1,3}(\d{1,3}){0,2}", self.text)
        for match in matches:
            if match:
                nums.append(TextContainer(self.group(), match.start(), match.end()))
        return matches

    def is_feet(self, offsets):
        if self.text[offsets[1]+1] == "\'":
            return True
        else:
            return False
