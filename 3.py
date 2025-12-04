// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © nbbnlmbm43

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// Join our channel for more free tools: https://t.me/simpleforextools

//@version=5
indicator("FT Concept v2.2"
  , shorttitle="FT v2.2"
  , overlay=true
  , max_lines_count = 100
  , max_boxes_count = 100
  , max_labels_count = 100
  , max_bars_back = 500
  )
 
///////////===========================Choch Settings===========================
showLine    	= input.bool(true, title="Show Line", group = "Line------------------------------------------")
///////////==========================================Management==========================================
manageArrayValue(arrSource, value) =>
	if arrSource.size() > 0
    	array.set(arrSource, 0, value)
	else
    	array.push(arrSource, value)
 
manageArrayLine(arrLine, value, maxNumb) =>
	if arrLine.size() > maxNumb
    	line.delete(array.shift(arrLine))
	array.push(arrLine, value)
 
manageArrayLable(arrLable, value, maxNumb) =>
	if arrLable.size() > maxNumb
    	label.delete(array.shift(arrLable))
	array.push(arrLable, value)
 
///////////===========================Main Variable===========================
var float[] __arrLow = array.new_float(na)
var float[] __arrHigh = array.new_float(na)
 
var line[] _arrLineHigh = array.new_line()
var line[] _arrLineLow = array.new_line()
 
var label[] _arrLable = array.new_label()
 
var bool	_continueFlag = false
var int 	_bias = na
 
isUpCandle   = close > open
isDownCandle = close < open
_pair_updown = close[1] > open[1] and close < open
_pair_downup = close[1] < open[1] and close > open
 
maxLine = 0
///////////===========================Main Logic===========================
if barstate.isconfirmed
	if not _continueFlag
    	if _pair_downup
        	_barLower = low < low[1] ? low : low[1]
        	_x1LowIndex = low < low[1] ? bar_index : bar_index[1]
        	manageArrayValue(__arrLow, _barLower)
 
        	if showLine
            	_drawLineBot = line.new(_x1LowIndex, _barLower, bar_index + 7, _barLower, color=color.red, width=1, style = line.style_solid)
            	manageArrayLine(_arrLineLow, _drawLineBot, maxLine)
 
	if array.size(__arrLow) > 0
    	// sweep low => move new low
    	_valLow = array.get(__arrLow, 0)
    	if low < _valLow and close - _valLow > 0 and close != low and array.size(__arrHigh) == 0
        	_barLower = low
        	_x1LowIndex = bar_index
        	manageArrayValue(__arrLow, _barLower)
 
        	if showLine
            	_drawLineBot = line.new(_x1LowIndex, _barLower, bar_index + 15, _barLower, color=color.red, width=1, style = line.style_dashed)
            	manageArrayLine(_arrLineLow, _drawLineBot, maxLine)
 
    	_latestLowVal = array.get(__arrLow, 0)
    	if _latestLowVal - close > 0 and isDownCandle
        	__arrTempHigh = array.new_float(4)
        	for i = 0 to 3
            	array.set(__arrTempHigh, i, high[i])
 
        	_lookupHighValue = __arrTempHigh.get(0)
 
        	for i = 1 to 3
            	if __arrTempHigh.get(i) > _lookupHighValue
                	_lookupHighValue	:= __arrTempHigh.get(i)
 
        	manageArrayValue(__arrHigh, _lookupHighValue)
        	_continueFlag := true
        	_bias := -1
    	
    	if _continueFlag
        	if _pair_updown
            	_barHigher   = high > high[1] ? high      : high[1]
            	_x1HighIndex = high > high[1] ? bar_index : bar_index[1]
            	manageArrayValue(__arrHigh, _barHigher)
 
            	if showLine
                	_drawLineTop = line.new(_x1HighIndex, _barHigher, bar_index + 7, _barHigher, color=color.blue, width=1, style = line.style_solid)
                	manageArrayLine(_arrLineHigh, _drawLineTop, maxLine)
 
    	if array.size(__arrHigh) > 0
        	_valHigh = array.get(__arrHigh, 0)
        	if high > _valHigh and close < _valHigh and high != open
            	_barHigher = high
            	_x1HighIndex = bar_index
            	manageArrayValue(__arrHigh, _valHigh)
 
            	if showLine
                	_drawLineTop = line.new(_x1HighIndex, _barHigher, bar_index + 15, _barHigher, color=color.blue, width=1, style = line.style_dashed)
                	manageArrayLine(_arrLineHigh, _drawLineTop, maxLine)
 
        	_latestHighVal = array.get(__arrHigh, 0)
        	if close - _latestHighVal > 0 and isUpCandle
            	__arrTempLow = array.new_float(4)
            	for i = 0 to 3
                	array.set(__arrTempLow, i, low[i])
 
            	_lookupLowVal = __arrTempLow.get(0)
 
            	for i = 1 to 3
                	if __arrTempLow.get(i) < _lookupLowVal
                    	_lookupLowVal	:= __arrTempLow.get(i)
 
            	manageArrayValue(__arrLow, _lookupLowVal)
            	
            	array.clear(__arrHigh)
            	_continueFlag := false
            	_bias := 1
 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////Check Bias Change + Alert/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
biasUpAlert 	= input.bool(true, "Alert Bias Up", group='Bias------------------------------------------')
biasDownAlert   = input.bool(true, "Alert Bias Down", group='Bias------------------------------------------')
showLabelBias  = input.bool(true, title="Bias Label", group = "Bias------------------------------------------")
 
__isFvgUp = low > high[2] and close[1] > high[2]
__isFvgDown = high < low[2] and close[1] < low[2]
//////////////////// Variable//////////////////
var bool bullBias = false
var bool bearBias = false
var int countBull = 0
var int countBear = 0
var label[] _arrBiasLable = array.new_label()
if ta.change(_continueFlag)
	countBull := 0
	countBear := 0
 
if __isFvgUp and _bias == 1
	if countBull == 0
    	bullBias := true
	countBull := countBull + 1
 
if __isFvgDown and _bias == -1
	if countBear == 0
    	bearBias := true
	countBear := countBear + 1
 
if bullBias and biasUpAlert
	if showLabelBias
    	_drawBiasLable = label.new(x = bar_index - 1, y = low, color = color.blue, style=label.style_triangleup, xloc=xloc.bar_index, yloc=yloc.belowbar, size = size.tiny)
    	manageArrayLable(_arrBiasLable, _drawBiasLable, 60)
	alert("🍏 " + str.tostring(syminfo.ticker) + ", 🕰" + str.tostring(timeframe.period) + "Min Bias Tăng" , alert.freq_once_per_bar_close)
	bullBias := false
 
if bearBias and biasDownAlert
	if showLabelBias
    	_drawBiasLable = label.new(x = bar_index - 1, y = high, color = color.red, style=label.style_triangledown, xloc=xloc.bar_index, yloc=yloc.abovebar, size = size.tiny)
    	manageArrayLable(_arrBiasLable, _drawBiasLable, 60)
	alert("🍎 " + str.tostring(syminfo.ticker) + ", 🕰 " + str.tostring(timeframe.period) + "Min Bias Giảm", alert.freq_once_per_bar_close)
	bearBias := false
 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////HTF FVG (Fadi v7)/////////////////////////////////////////////////////////////////////
// © fadizeidan is the author of HTF FVG
//+------------------------------------------------------------------------------------------------------------+//
//+--- Types                                                                                                ---+//
//+------------------------------------------------------------------------------------------------------------+//
type Settings
	bool    CE_show
	string  CE_style
	color   CE_color
	bool    Border_show
	bool    mitigated_show
	string  mitigated_type
	color   mitigated_color_bull
	color   mitigated_color_bear
	bool    ltf_hide
	bool    label_show
	color   label_color
	color   label_bgcolor
	string  label_size
	int     padding
	int     buffer
 
type Imbalance_Settings
	bool    show
	string  htf
	color   color_bull
	color   color_bear
	int     max_count
 
type Imbalance
	int     open_time
	int     close_time
	float   open
	float   middle
	float   close
	bool    mitigated
	int     mitigated_time
	line    line_middle
	label   lbl
	box     box
 
type ImbalanceStructure
	Imbalance[]     	imbalance
	Imbalance_Settings  settings
 
type Helper
	string name     	= "Helper"
 
//+------------------------------------------------------------------------------------------------------------+//
//+--- Settings                                                                                             ---+//
//+------------------------------------------------------------------------------------------------------------+//
Settings_Group                  	= "HTF FVGs Settings"
 
Settings settings               	= Settings.new()
Imbalance_Settings HTF_1_Settings   = Imbalance_Settings.new()
Imbalance_Settings HTF_2_Settings   = Imbalance_Settings.new()
Imbalance_Settings HTF_3_Settings   = Imbalance_Settings.new()
Imbalance_Settings HTF_4_Settings   = Imbalance_Settings.new()
Imbalance_Settings HTF_5_Settings   = Imbalance_Settings.new()
Imbalance_Settings HTF_6_Settings   = Imbalance_Settings.new()
 
string tooltip1 = "HTF FVG Settings:\n\tShow/Hide timeframe\n\tTimeframe to display\n\tBullish FVG Color\n\tBearish FVG Color\n\tMaximum number of FVGs to display"
string tooltip2 = "Mitigated FVG Settings:\n\tShow/Hide mitigated (Applies to all).\n\tBullish FVG Color\n\tBearish FVG Color\n\tWhen to mark it as mitigated (Based on HTF timeframe, not current timeframe)"
 
HTF_1_Settings.show     	:= input.bool(true, "", inline="htf1", group=Settings_Group)
htf_1                    	= input.timeframe("5", "", inline="htf1", group=Settings_Group)
HTF_1_Settings.htf      	:= htf_1
HTF_1_Settings.color_bull   := input.color(color.new(color.green,90), "", inline="htf1", group=Settings_Group)
HTF_1_Settings.color_bear   := input.color(color.new(color.blue,90), "", inline="htf1", group=Settings_Group)
HTF_1_Settings.max_count	:= input.int(20, "", inline="htf1", tooltip = tooltip1, group=Settings_Group)
 
HTF_2_Settings.show     	:= input.bool(true, "", inline="htf2", group=Settings_Group)
htf_2                    	= input.timeframe("15", "", inline="htf2", group=Settings_Group)
HTF_2_Settings.htf      	:= htf_2
HTF_2_Settings.color_bull   := input.color(color.new(color.green,90), "", inline="htf2", group=Settings_Group)
HTF_2_Settings.color_bear   := input.color(color.new(color.blue,90), "", inline="htf2", group=Settings_Group)
HTF_2_Settings.max_count	:= input.int(20, "", inline="htf2", tooltip = tooltip1, group=Settings_Group)
 
HTF_3_Settings.show     	:= input.bool(true, "", inline="htf3", group=Settings_Group)
htf_3                    	= input.timeframe("60", "", inline="htf3", group=Settings_Group)
HTF_3_Settings.htf      	:= htf_3
HTF_3_Settings.color_bull   := input.color(color.new(color.green,90), "", inline="htf3", group=Settings_Group)
HTF_3_Settings.color_bear   := input.color(color.new(color.blue,90), "", inline="htf3", group=Settings_Group)
HTF_3_Settings.max_count	:= input.int(20, "", inline="htf3", tooltip = tooltip1, group=Settings_Group)
 
settings.ltf_hide           	:= input.bool(true, "Hide Lower Timeframes", group=Settings_Group)
 
settings.Border_show        	:= input.bool(true, "Show Border", group=Settings_Group, inline="4")
settings.mitigated_show     	:= input.bool(true, "Show Mitigated", group=Settings_Group, inline="2")
settings.mitigated_color_bull   := input.color(color.new(color.gray,95), "", group=Settings_Group, inline="2")
settings.mitigated_color_bear   := input.color(color.new(color.gray,95), "", group=Settings_Group, inline="2")
settings.mitigated_type     	:= input.string('Wick filled', 'when', options = ['None', 'Wick filled', 'Body filled', 'Wick filled half', 'Body filled half'], group=Settings_Group, inline="2", tooltip=tooltip2)
settings.CE_show            	:= input.bool(true, "Show C.E.     ", group=Settings_Group, inline="3")
settings.CE_color           	:= input.color(color.new(color.black,60), "", group=Settings_Group, inline="3")
settings.CE_style           	:= input.string('····', '     ', options = ['⎯⎯⎯', '----', '····'], group=Settings_Group, inline="3")
settings.label_show         	:= input.bool(true, "Show Label   ", inline="label")
settings.label_color        	:= input.color(color.new(color.black, 10), "", inline='label')
settings.label_bgcolor      	:= input.color(color.new(color.white, 100), "", inline='label')
settings.label_size         	:= input.string(size.small, "", [size.tiny, size.small, size.normal, size.large, size.huge], inline="label")
settings.padding            	:= input.int(4, "Distance from current candle", minval=0)
settings.buffer             	:= input.int(6, "Spacing between timeframes", minval=0)
 
//+------------------------------------------------------------------------------------------------------------+//
//+--- Variables                                                                                            ---+//
//+------------------------------------------------------------------------------------------------------------+//
color color_transparent         	= #ffffff00
 
Helper helper                   	= Helper.new()
 
var ImbalanceStructure FVG_1    	= ImbalanceStructure.new()
var ImbalanceStructure FVG_2    	= ImbalanceStructure.new()
var ImbalanceStructure FVG_3    	= ImbalanceStructure.new()
 
var Imbalance[] FVGs_1          	= array.new<Imbalance>()
var Imbalance[] FVGs_2          	= array.new<Imbalance>()
var Imbalance[] FVGs_3          	= array.new<Imbalance>()
 
FVG_1.imbalance                 	:= FVGs_1
FVG_1.settings                  	:= HTF_1_Settings
FVG_2.imbalance                 	:= FVGs_2
FVG_2.settings                  	:= HTF_2_Settings
FVG_3.imbalance                 	:= FVGs_3
FVG_3.settings                  	:= HTF_3_Settings
 
//Used internally for padding
var int TF_1    	= 0
var int TF_2    	= 0
var int TF_3    	= 0
 
var float daily 	= 0
var float monthly   = 0
//+------------------------------------------------------------------------------------------------------------+//
//+--- Methods                                     	                                                     ---+//
//+------------------------------------------------------------------------------------------------------------+//
method LineStyle(Helper helper, string style) =>
	helper.name := style
 
	out = switch style
    	'----' => line.style_dashed
    	'····' => line.style_dotted
    	=> line.style_solid
	
	out
 
method Gethtftext(Helper helper, string htf) =>
	helper.name := htf
 
	formatted = htf
	seconds = timeframe.in_seconds(htf)
 
	if seconds < 60
    	formatted := str.tostring(seconds) + "s"
	else if (seconds / 60) < 60
    	formatted := str.tostring((seconds/60)) + "m"
	else if (seconds/60/60) < 24
    	formatted := str.tostring((seconds/60/60)) + "H"
	
	formatted
 
method Validtimeframe(Helper helper, tf) =>
	helper.name := tf
	n1 = timeframe.in_seconds()
	n2 = timeframe.in_seconds(tf)
 
	n1 < n2
 
method ProximityRange(Helper helper, tf) =>
	helper.name := tf
	float range_high	= 0
	float range_low   = 0
	if timeframe.isseconds or timeframe.isminutes
    	range_high := close + daily
    	range_low  := close - daily
	if timeframe.isdaily
    	range_high := close + monthly*3
    	range_low  := close - monthly*3
	if timeframe.isweekly
    	range_high := close + monthly*12
    	range_low  := close - monthly*12
 
	[range_low, range_high]
 
//+------------------------------------------------------------------------------------------------------------+//
//+--- Imbalances Methods                                                                                   ---+//
//+------------------------------------------------------------------------------------------------------------+//
// AddZone is used to display and manage imbalance related boxes
method AddZone(ImbalanceStructure IS, Imbalance imb, int step) =>
	bool visible = true
	if IS.settings.show
    	int buffer = time+((time-time[1])*(settings.padding+1+(settings.buffer*(step-1))))
 
    	if na(imb.box)
        	imb.box := box.new(imb.open_time, imb.open, buffer, imb.close, settings.Border_show ? imb.open < imb.close ? color.new(IS.settings.color_bull, color.t(IS.settings.color_bull)/ 3) : color.new(IS.settings.color_bear, color.t(IS.settings.color_bear)/ 3) : color_transparent, settings.Border_show ? 1 : 0, bgcolor = imb.open < imb.close ? IS.settings.color_bull : IS.settings.color_bear, xloc=xloc.bar_time)
        	if settings.label_show
            	imb.lbl := label.new(buffer, imb.middle, text=helper.Gethtftext(IS.settings.htf), xloc=xloc.bar_time, textcolor=settings.label_color, style=label.style_label_left, color=settings.label_bgcolor, size = settings.label_size)
        	if settings.CE_show
            	imb.line_middle := line.new(imb.open_time, imb.middle, buffer, imb.middle, xloc=xloc.bar_time, style=helper.LineStyle(settings.CE_style), color=settings.CE_color)
    	else
        	box.set_right(imb.box, imb.mitigated ? imb.mitigated_time : buffer)
        	box.set_bgcolor(imb.box, imb.open < imb.close ? imb.mitigated ? settings.mitigated_color_bull : IS.settings.color_bull : imb.mitigated ? settings.mitigated_color_bear : IS.settings.color_bear)
        	box.set_border_color(imb.box, settings.Border_show ? imb.open < imb.close ? color.new(settings.mitigated_color_bull, color.t(settings.mitigated_color_bull)/ 3) : color.new(settings.mitigated_color_bear, color.t(settings.mitigated_color_bear)/ 3) : color_transparent)
        	label.set_x(imb.lbl, imb.mitigated ? imb.mitigated_time : buffer)
        	//label.delete(imb.lbl)
        	if settings.CE_show
            	line.set_x2(imb.line_middle, imb.mitigated ? imb.mitigated_time : buffer)
    	if imb.mitigated and not settings.mitigated_show
        	if not na(imb.box)
            	box.delete(imb.box)
            	line.delete(imb.line_middle)
            	label.delete(imb.lbl)
            	visible := false
	visible
 
// AddImbalance adds a newly discovered imbalance. this applies for both FVG and Volume Imbalance
method AddImbalance(ImbalanceStructure IS, float o, float c, int o_time) =>
	Imbalance imb = Imbalance.new()
	imb.open_time       	:= o_time
	imb.open            	:= o
	imb.middle          	:= (o+c)/2
	imb.close           	:= c
 
	IS.imbalance.unshift(imb)
	//IS.AddZone(imb)
 
	if IS.imbalance.size() > 100 //IS.settings.max_count
    	temp = IS.imbalance.pop()
    	if not na(temp.box)
        	box.delete(temp.box)
        	line.delete(temp.line_middle)
        	label.delete(temp.lbl)
	IS
 
// CheckMitigated checks if the imbalance has been mitigated based on the settings
method CheckMitigated(ImbalanceStructure IS, o, h, l, c) =>
	if IS.imbalance.size() > 0
    	for i = IS.imbalance.size() - 1 to 0
        	imb = IS.imbalance.get(i)
 
        	if not imb.mitigated
            	switch settings.mitigated_type
                	"None" =>
                    	imb.mitigated   	:= false
                	'Wick filled' =>
                    	imb.mitigated   	:= imb.open <= imb.close ? low <= imb.open : high >= imb.open
                	'Body filled' =>
                    	imb.mitigated   	:= imb.open < imb.close ? math.min(o, c) <= imb.open : math.max(o, c) >= imb.open
                	'Wick filled half' =>
                    	imb.mitigated   	:= imb.open <= imb.close ? low <= imb.middle : high >= imb.middle
                	'Body filled half' =>
                    	imb.mitigated   	:= imb.open <= imb.close ? math.min(o, c) <= imb.middle : math.max(o, c) >= imb.middle
            	if imb.mitigated
                	if not settings.mitigated_show
                    	if not na(imb.box)
                        	box.delete(imb.box)
                        	line.delete(imb.line_middle)
                        	label.delete(imb.lbl)
                    	IS.imbalance.remove(i)
                	else
                    	imb.mitigated_time  := time
	IS
 
method AdjustMargins(ImbalanceStructure IS, int step) =>
	int count = 0
	if IS.imbalance.size() > 0
    	int buffer = time+((time-time[1])*(settings.padding+1+(settings.buffer*(step-1))))
    	
    	[rl, rh] = helper.ProximityRange(IS.settings.htf)
 
    	for i = 0 to IS.imbalance.size() - 1
        	imb = IS.imbalance.get(i)
        	if ((math.max(imb.open, imb.close) > rl) and (math.min(imb.open, imb.close) < rh)) and count < IS.settings.max_count
            	if IS.AddZone(imb, step)
                	count := count+1
        	else
            	if not na(imb.box)
                	box.delete(imb.box)
                	label.delete(imb.lbl)
                	line.delete((imb.line_middle))
	IS
// FindImbalance looks for imbalances and, if found, adds it to the list
method FindImbalance(ImbalanceStructure IS, o, h, l, c, t, o1, h1, l1, c1, t1, o2, h2, l2, c2, t2) =>
	if IS.settings.show and (h < l2 or l > h2)
    	o = h < l2 ? l2 : h2
    	c = h < l2 ? h : l
    	if IS.imbalance.size() == 0
        	IS.AddImbalance(o, c, t2)
    	else
        	if IS.imbalance.first().open_time < t2
            	IS.AddImbalance(o, c, t2)
	IS
 
method Process(ImbalanceStructure IS, float o, float h, float l, float c, int t, float o1, float h1, float l1, float c1, int t1, float o2, float h2, float l2, float c2, int t2) =>
	var int visible = 0
	if IS.settings.show
    	if not settings.ltf_hide or (settings.ltf_hide and helper.Validtimeframe(IS.settings.htf))
        	if IS.settings.show
            	IS.FindImbalance(o, h, l, c, t, o1, h1, l1, c1, t1, o2, h2, l2, c2, t2)
            	visible := 1
    	IS.CheckMitigated(o, h, l, c)
	visible
//+------------------------------------------------------------------------------------------------------------+//
//+--- Main call to start the process                                                                       ---+//
//+------------------------------------------------------------------------------------------------------------+//
 
daily                               	:= request.security(syminfo.tickerid, "1D", ta.atr(14))
monthly                             	:= request.security(syminfo.tickerid, "1M", ta.atr(14))
 
[o_1, h_1, l_1, c_1, t_1]           	= request.security(syminfo.tickerid, htf_1, [open[1], high[1], low[1], close[1], time[1]])
[o1_1, h1_1, l1_1, c1_1, t1_1]      	= request.security(syminfo.tickerid, htf_1, [open[2], high[2], low[2], close[2], time[2]])
[o2_1, h2_1, l2_1, c2_1, t2_1]      	= request.security(syminfo.tickerid, htf_1, [open[3], high[3], low[3], close[3], time[3]])
TF_1 := FVG_1.Process(o_1, h_1, l_1, c_1, t_1, o1_1, h1_1, l1_1, c1_1, t1_1, o2_1, h2_1, l2_1, c2_1, t2_1)
FVG_1.AdjustMargins(TF_1)
 
[o_2, h_2, l_2, c_2, t_2]           	= request.security(syminfo.tickerid, htf_2, [open[1], high[1], low[1], close[1], time[1]])
[o1_2, h1_2, l1_2, c1_2, t1_2]      	= request.security(syminfo.tickerid, htf_2, [open[2], high[2], low[2], close[2], time[2]])
[o2_2, h2_2, l2_2, c2_2, t2_2]      	= request.security(syminfo.tickerid, htf_2, [open[3], high[3], low[3], close[3], time[3]])
TF_2 := TF_1 + FVG_2.Process(o_2, h_2, l_2, c_2, t_2, o1_2, h1_2, l1_2, c1_2, t1_2, o2_2, h2_2, l2_2, c2_2, t2_2)
FVG_2.AdjustMargins(TF_2)
 
[o_3, h_3, l_3, c_3, t_3]           	= request.security(syminfo.tickerid, htf_3, [open[1], high[1], low[1], close[1], time[1]])
[o1_3, h1_3, l1_3, c1_3, t1_3]      	= request.security(syminfo.tickerid, htf_3, [open[2], high[2], low[2], close[2], time[2]])
[o2_3, h2_3, l2_3, c2_3, t2_3]      	= request.security(syminfo.tickerid, htf_3, [open[3], high[3], low[3], close[3], time[3]])
TF_3 := TF_2 + FVG_3.Process(o_3, h_3, l_3, c_3, t_3, o1_3, h1_3, l1_3, c1_3, t1_3, o2_3, h2_3, l2_3, c2_3, t2_3)
FVG_3.AdjustMargins(TF_3)
 
//Settings
show_fvg = input(true, 'Fair Value Gaps (FVG)', inline = 'fvg_css', group = 'LuxAlgo Imbalance')
bull_fvg_css = input.color(#2157f3, '', inline = 'fvg_css', group = 'LuxAlgo Imbalance')
bear_fvg_css = input.color(#ff1100, '', inline = 'fvg_css', group = 'LuxAlgo Imbalance')
fvg_usewidth = input(false, 'Min Width', inline = 'fvg_width', group = 'LuxAlgo Imbalance')
fvg_gapwidth = input.float(0, '', inline = 'fvg_width', group = 'LuxAlgo Imbalance')
fvg_method = input.string('Points', '', options = ['Points', '%', 'ATR'], inline = 'fvg_width', group = 'LuxAlgo Imbalance')
fvg_extend = input.int(0, 'Extend FVG', minval = 0, group = 'LuxAlgo Imbalance')

n_lux = bar_index
atr = ta.atr(200)
imbalance_detection_lux(show, usewidth, method, width, top, btm, condition)=>
    var is_width = true
    var count = 0
    if usewidth
        dist = top - btm
        is_width := switch method
            'Points' => dist > width
            '%' => dist / btm * 100 > width
            'ATR' => dist > atr * width
    is_true = show and condition and is_width
    count += is_true ? 1 : 0
    [is_true, count]

bull_filled_lux(condition, btm)=>
    var btms = array.new_float(0)
    var count = 0
    if condition
        array.unshift(btms, btm)
    size = array.size(btms)
    for i = (size > 0 ? size-1 : na) to 0
        value = array.get(btms, i)
        if low < value
            array.remove(btms, i)
            count += 1
    count

bear_filled_lux(condition, top)=>
    var tops = array.new_float(0)
    var count = 0
    if condition
        array.unshift(tops, top)
    size = array.size(tops)
    for i = (size > 0 ? size-1 : na) to 0
        value = array.get(tops, i)
        if high > value
            array.remove(tops, i)
            count += 1
    count

[bull_fvg, bull_fvg_count] = imbalance_detection_lux(show_fvg, fvg_usewidth, fvg_method, fvg_gapwidth, low, high[2], low > high[2] and close[1] > high[2])
bull_fvg_filled = bull_filled_lux(bull_fvg, high[2])
[bear_fvg, bear_fvg_count] = imbalance_detection_lux(show_fvg, fvg_usewidth, fvg_method, fvg_gapwidth, low[2], high, high < low[2] and close[1] < low[2])
bear_fvg_filled = bear_filled_lux(bear_fvg, low[2])

if bull_fvg
    avg = math.avg(low, high[2])
    box.new(n_lux-2, low, n_lux + fvg_extend, high[2], border_color = na, bgcolor = color.new(bull_fvg_css, 80))
    line.new(n_lux-2, avg, n_lux + fvg_extend, avg, color = bull_fvg_css)
if bear_fvg
    avg = math.avg(low[2], high)
    box.new(n_lux-2, low[2], n_lux + fvg_extend, high, border_color = na, bgcolor = color.new(bear_fvg_css, 80))
    line.new(n_lux-2, avg, n_lux + fvg_extend, avg, color = bear_fvg_css)
// -----------------------------------------------------------------------------}
// -----------------------------------------------------------------------------
// SB CISD Detector Section
// -----------------------------------------------------------------------------{
space = '                               '
Tracking_Method = input.string('Classic', 'Detection Method', options=['Classic', 'Liquidity Sweep'], group = 'SB CISD Detector')
length = input.int(10, 'Swing Length', minval=1, group = 'SB CISD Detector')
Minimum_Sequence_Length = input.int(0, 'Minimum CISD Duration', minval=0, group = 'SB CISD Detector')
Maximum_Sequence_Length = input.int(100, 'Maximum Swing Validity', minval=1, maxval=1000, group = 'SB CISD Detector')
textSize = str.lower(input.string('Tiny', 'Label/Text Size', group='SB CISD Style', options=['Tiny', 'Small', 'Normal', 'Large']))
lineWidth = input.int(3, 'Line/Dash Width', minval=1, maxval=10, group='SB CISD Style')
cBull_sb = input.color(#089981, 'Bullish', group='SB CISD Style')
cBear_sb = input.color(#f23645, 'Bearish', group='SB CISD Style')
cSweepH = input.color(#787b8684, 'Sweeps' + space, group='SB CISD Style', inline="S")
cSweepL = input.color(#787b8684, '', group='SB CISD Style', inline="S")

type bin 
    line ln 
    bool active 
    chart.point cp1
    chart.point cp2
    bool broken = false

type swing 
    chart.point cp
    line ln
    line wick
    bool active = false

INV = color(na)
n_sb = bar_index 
bull_sb = close > open 
bear_sb = close < open 
sweep = Tracking_Method == 'Liquidity Sweep'

var int trend = 0
var array<bin> arrBull = array.new<bin>() 
var array<bin> arrBear = array.new<bin>() 
var array<swing> swingsH = array.new<swing>() 
var array<swing> swingsL = array.new<swing>() 
var chart.point cp_lastPh = chart.point.from_index(n_sb, high) 
var chart.point cp_lastPl = chart.point.from_index(n_sb, low) 
var chart.point trackPriceBull = chart.point.from_index(na, na)
var chart.point trackPriceBear = chart.point.from_index(na, na)

var bin oBull = bin.new(line.new(n_sb, open, n_sb, open, color=color.green, width=lineWidth),true,chart.point.from_index(n_sb, high),chart.point.from_index(n_sb, high))
var bin oBear = bin.new(line.new(n_sb, open, n_sb, open, color=color.red, width=lineWidth),true,chart.point.from_index(n_sb, low),chart.point.from_index(n_sb, low))

ph = ta.pivothigh(length, 1)
pl = ta.pivotlow(length, 1)

if not na(ph)
    swingsH.push(swing.new(chart.point.from_index(n_sb - 1, ph)))
    cp_lastPh := chart.point.from_index(n_sb - 1, ph)

if not na(pl)
    swingsL.push(swing.new(chart.point.from_index(n_sb - 1, pl)))
    cp_lastPl := chart.point.from_index(n_sb - 1, pl)

bullishCISD = false
bearishCISD = false

if not sweep
    if bull_sb and bear_sb[1]
        trackPriceBull := chart.point.from_index(n_sb, open)
        if oBull.active and not oBull.broken
            oBull.ln.delete()
        oBull.active := true 
        oBull.ln := line.new(trackPriceBull.index, trackPriceBull.price, n_sb, trackPriceBull.price, color=color.new(cBull_sb, 40), width=lineWidth)
        oBull.cp1 := chart.point.from_index(cp_lastPh.index, cp_lastPh.price)        
        oBull.cp2 := chart.point.from_index(n_sb, cp_lastPh.price)
        oBull.broken := false

    if bear_sb and bull_sb[1] 
        trackPriceBear := chart.point.from_index(n_sb, open)
        if oBear.active and not oBear.broken
            oBear.ln.delete()
        oBear.active := true 
        oBear.ln := line.new(trackPriceBear.index, trackPriceBear.price, n_sb, trackPriceBear.price, color=color.new(cBear_sb, 40), width=lineWidth)
        oBear.cp1 := chart.point.from_index(cp_lastPl.index, cp_lastPl.price)        
        oBear.cp2 := chart.point.from_index(n_sb, cp_lastPl.price)
        oBear.broken := false

    if oBull.active
        if n_sb - oBull.ln.get_x1() <= Maximum_Sequence_Length
            oBull.ln.set_x2(n_sb)
            if close < oBull.ln.get_y2()
                if n_sb - oBull.ln.get_x1() >= Minimum_Sequence_Length
                    oBull.ln.set_color(cBear_sb)
                    if trend == -1
                        oBull.ln.set_style(line.style_dashed)
                        oBull.ln.set_width(lineWidth)
                    oBull.active := false
                    oBull.broken := true
                    trend := -1
                    x = math.ceil(math.avg(oBull.ln.get_x1(), n_sb))
                    label.new(x, oBull.ln.get_y2(), style=label.style_label_up, textcolor=cBear_sb, text='CISD', size=textSize, color=INV)
                    bearishCISD := true
                else 
                    oBull.active := false
                    oBull.ln.delete()
        else 
            oBull.active := false
            oBull.ln.delete()

    if oBear.active
        if n_sb - oBear.ln.get_x1() <= Maximum_Sequence_Length
            oBear.ln.set_x2(n_sb)
            if close > oBear.ln.get_y2()
                if n_sb - oBear.ln.get_x1() >= Minimum_Sequence_Length
                    oBear.ln.set_color(cBull_sb)
                    if trend == 1
                        oBear.ln.set_style(line.style_dashed)
                        oBear.ln.set_width(lineWidth)
                    oBear.active := false
                    oBear.broken := true
                    trend := 1
                    x = math.ceil(math.avg(oBear.ln.get_x1(), n_sb))
                    label.new(x, oBear.ln.get_y2(), style=label.style_label_down, textcolor=cBull_sb, text='CISD', color=INV, size=textSize)
                    bullishCISD := true
                else
                    oBear.active := false
                    oBear.ln.delete()
        else 
            oBear.active := false
            oBear.ln.delete()
// -----------------------------------------------------------------------------}
// -----------------------------------------------------------------------------
// ote618 iFVG Detector Section
// -----------------------------------------------------------------------------{
alert_tf = input.string("All", title="Enable Alerts For Timeframe", options=["All", "1", "5", "15", "60"], group = 'iFVG Detector')
current_tf = timeframe.period
should_alert = (alert_tf == "All") or (alert_tf == current_tf)

var bool ig_active = false
var int ig_start_bar = na
var int ig_detected_bar = na
var float ig_c1_high = na
var float ig_c1_low = na
var float ig_c3_high = na
var float ig_c3_low = na
var int ig_direction = 0
var int ig_validation_end = na
var box ig_box = na

bullishFVG = high[2] < low
bearishFVG = low[2] > high

if bullishFVG
    ig_active := true
    ig_direction := 1
    ig_start_bar := bar_index - 2
    ig_detected_bar := bar_index
    ig_c1_high := high[2]
    ig_c1_low := low[2]
    ig_c3_high := high
    ig_c3_low := low
    ig_validation_end := bar_index + 4

if bearishFVG
    ig_active := true
    ig_direction := -1
    ig_start_bar := bar_index - 2
    ig_detected_bar := bar_index
    ig_c1_high := high[2]
    ig_c1_low := low[2]
    ig_c3_high := high
    ig_c3_low := low
    ig_validation_end := bar_index + 4

validated = false
if ig_active and bar_index <= ig_validation_end
    if ig_direction == 1 and close < ig_c1_high
        validated := true
    if ig_direction == -1 and close > ig_c1_low
        validated := true

    if validated
        top = ig_direction == 1 ? ig_c1_high : ig_c3_high
        bottom = ig_direction == 1 ? ig_c3_low : ig_c1_low
        ig_box := box.new(left=ig_start_bar,right=bar_index,top=top,bottom=bottom,bgcolor=color.new(ig_direction == 1 ? #FF073A : #319120,30), border_color=color.rgb(120, 123, 134, 100))
        if should_alert
            direction_txt = ig_direction == 1 ? "Bearish" : "Bullish"
            alert(direction_txt + " iFVG confirmed on " + current_tf, alert.freq_once_per_bar_close)
        ig_active := false
// -----------------------------------------------------------------------------}
// -----------------------------------------------------------------------------
// Alert Toggles Section
// -----------------------------------------------------------------------------{
alert_group = 'Alert Toggles'
alert_fvg_bull = input(true, 'Bullish FVG', group=alert_group, inline='fvg_alerts')
alert_fvg_bear = input(true, 'Bearish FVG', group=alert_group, inline='fvg_alerts')
alert_cisd_bull = input(true, 'Bullish CISD', group=alert_group, inline='cisd_alerts')
alert_cisd_bear = input(true, 'Bearish CISD', group=alert_group, inline='cisd_alerts')
alert_ifvg_bull = input(true, 'Bullish iFVG', group=alert_group, inline='ifvg_alerts')
alert_ifvg_bear = input(true, 'Bearish iFVG', group=alert_group, inline='ifvg_alerts')

// LuxAlgo Alerts
alertcondition(bull_fvg and alert_fvg_bull, 'Bullish FVG Alert', 'A Bullish FVG event detected.')
if bull_fvg and alert_fvg_bull
    alert("🍏 Bullish FVG detected on " + current_tf, alert.freq_once_per_bar_close)
alertcondition(bear_fvg and alert_fvg_bear, 'Bearish FVG Alert', 'A Bearish FVG event detected.')
if bear_fvg and alert_fvg_bear
    alert("🍎 Bearish FVG detected on " + current_tf, alert.freq_once_per_bar_close)

// SB Alerts
alertcondition(bullishCISD and alert_cisd_bull, 'Bullish CISD Alert', 'A Bullish CISD event detected (Instant).')
if bullishCISD and alert_cisd_bull
    alert("🍏 Bullish CISD Detected [Instant] on " + current_tf, alert.freq_once_per_bar_close)
alertcondition(bearishCISD and alert_cisd_bear, 'Bearish CISD Alert', 'A Bearish CISD event detected (Instant).')
if bearishCISD and alert_cisd_bear
    alert("🍎 Bearish CISD Detected [Instant] on " + current_tf, alert.freq_once_per_bar_close)

// ote618 Alerts
if validated and should_alert
    direction_txt = ig_direction == 1 ? "Bearish" : "Bullish"
    emoji = ig_direction == 1 ? "🍎" : "🍏"
    if ig_direction == 1 and alert_ifvg_bear
        alert(emoji + " " + direction_txt + " iFVG confirmed on " + current_tf, alert.freq_once_per_bar_close)
    if ig_direction == -1 and alert_ifvg_bull
        alert(emoji + " " + direction_txt + " iFVG confirmed on " + current_tf, alert.freq_once_per_bar_close)


// ==========================================================================================

// === Dashboard with Telegram Link ===
var table myTable = table.new(position.top_center, 1, 1, border_width=1, frame_color=color.black, bgcolor=color.white)

// Add Telegram Message to Dashboard
table.cell(myTable, 0, 0, "Join Telegram @simpleforextools", bgcolor=color.blue, text_color=color.white, text_size=size.normal)