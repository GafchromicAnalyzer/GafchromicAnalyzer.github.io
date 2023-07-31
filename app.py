# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
 
from cmath import nan
from operator import contains
import sys
from dash import Dash, html, dcc, Input, Output,State, callback_context
import dash
import dash_daq as daq
from PIL import Image,ImageOps, ImageDraw
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

from plotly.io import write_image
import webbrowser
import base64
from io import BytesIO
import math
import peakutils
from scipy.optimize import OptimizeWarning
import warnings
import plotly.io as pio



# https://community.plotly.com/t/dash-range-slider-which-able-to-interact-with-input-field-that-display-range-slider-value/49476/3
# https://pages.github.com/
# https://github.com/covid19-dash/covid-dashboard/blob/master/Makefile

''' Though this code is made in python there still needs to be html elements to have a front end display of the data. id's are refrences to the varirables and 
styles help with the visual layout of the page. In addition to this are some parameters that I predefined, either as they don't impact callbacks or to make the
null image layout look nicer.

'''
 
app = Dash(__name__) 

app.layout = html.Div(children=[
    html.Div(children=[
        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='3D Model', children=[html.Div(children=[ #1st tab
                html.Div(children=[dcc.Graph(id='3_dimensional_graph', style={'width':'140vh','height':'90vh'})]), #3d graph
                html.Div(children=[
                    html.H5(id = "uploaded_filename"),
                    html.Div(children=[html.Img(id='image_uploaded', style={'display':'block','margin-left':'auto','margin-right':'auto','width':'70%'})]), #display of image
                    dcc.Upload(
                        id='upload',
                        children=html.Div(['Click to ', html.A('Select an Image')]),
                        style={'color': 'darkgray','width': '100%','height': '50px','lineHeight': '50px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','borderColor': 'darkgray','textAlign': 'center','padding': '2rem 0','margin-bottom': '2rem'},
                        accept='image/*',
                    ), #upload of image
                    html.Div(children=['Start Value:',dcc.Input(id='color_slider_start', type='number',min=0,debounce=True)]), # 3d color sliders 
                    html.Div(children=['End Value:',dcc.Input(id='color_slider_end', type='number',max=255,debounce=True)]), # 3d color sliders
                    dcc.Graph(
                        id='graph_histogram_colors',
                        config={'displayModeBar': False})
                    ], #3d graph color histogram
                    style={'width':'20%'}
                    )    
                ],
                style={'display': 'flex','justify-content': 'space-evenly','align-items': 'flex-start'})
            ]),
        dcc.Tab(label='ROI Selection', children=[ #2nd tab
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(id='ROI_graph'), #ROI graph
                ],style={'display':'flex','flex-direction':'column','align-items':'stretch'}),
                html.Div(children=[
                    html.H1('Selection Mode'), #text
                    dcc.RadioItems(id='selection_shape',options=['Circle', 'Rectangle'], value='Circle', inline=True), #buttons to select shape of ROI
                    html.Button('See', id='see_ROI_selection'), #See ROI shape
                    html.Button('Crop', id='crop_ROI_selection'), #Crop (also places result)
                    html.Button('Restart', id='restart_ROI_selection'), #Restart selection
                    html.H3('Std dev: ', id='standard_deviation'), #standard deviation result
                    html.H3('Uniformity: ', id='uniformity'), #uniformity result
                    dcc.Graph(id='background_subtration_graph'), #background subtraction graph
                    html.Button('calculate background subtraction', id ='calculate_background_subtraction'), # background subtraction button
                    html.H2('Background Subtracted: ', id='background_subtraction_output'), #backround subtraction number display
                ],style={'width':'20%'})
                ],style={'display': 'flex','justify-content': 'space-evenly','align-items': 'flex-start'})
                ]),
        dcc.Tab(label='Additional Graphs', children=[ #3rd tab
            dcc.Tabs(id='additional_graphs_tabs',children=[
                dcc.Tab(label='X Y plots', children=[html.Div(children=[ #3rd - 1st tab
                    html.Div(children=[
                        html.Div(children=[
                            dcc.Slider(id='yaxis_slider',min=0,value=0, max =100, vertical=True), #Y axis slider
                            html.Div(children=[
                                dcc.Graph(id='xy_graph',style={'width':'65vh','height':'65vh'}), # XY graph (basically the image)
                                dcc.Slider(id='xaxis_slider',min=0,value=0, max =100)],style={'width':'65vh','height':'65vh'} #X axis slider
                            )
                        ],style={'display': 'flex'}),
                        html.Div(children=[
                            dcc.Graph(id='yaxis_histogram',style={'width':'100vh','height':'30vh'}), #graph of y axis
                            dcc.Graph(id='xaxis_histogram',style={'width':'100vh','height':'30vh'}), #graph of x axis
                        ])
                        ],style={'display': 'flex'})
                        ]),
                ]),
                dcc.Tab(label='Contour Map', children=[html.Div(children=[ #3rd - 2nd tab
                        html.Div(children=[
                            dcc.Graph(id='contour_graph'), # graph of contour
                            html.H4('Tolerance:'), 
                            dcc.Input(id='contour_range',list='tolerance_range',type='text',style={'width':'20%'}), #tolerance 
                            html.Datalist(id='tolerance_range',children=[ # options
                            html.Option(value= '1', id='option_1'),html.Option(value= '2', id='option_2'),html.Option(value= '3', id='option_3')]),
                            html.Button('Percent', id='tolerance_type', n_clicks=0,style={'width':'20%'}), #change between % and stdev
                            ],style={'display': 'flex','align-items':'center','flex-direction':'column'})
                            ]),
                ])
            ])
        ]),
        dcc.Tab(label='Print PDF', children=[ #4th tab
                html.Div(children=[
                    html.Div(children=[
                        html.H5(id = "uploaded_filename_out"), #filename
                        html.Img(id='img_out',style= {'display':'block','margin-left':'auto','margin-right':'auto','width':'30%'}), #image
                        dcc.Graph(id='3d_graph_out',style= {'width':'30%'}), #3d graph
                        dcc.Graph('contour_graph_out',style= {'width':'30%'}) #contour graph
                        ]),
                    html.Div(children=[
                        dcc.Graph(id ='stdev_graph_out'), #ROI graph
                        html.H3('', id='Stdev_out'), #stdev number
                        html.H3('',id='uniformity_out'), #uniformity number
                        html.H3('', id='bac_sub_out') #bac_sub_out
                        
                    ]),
                    
                    ],
                style={'display': 'flex','justify-content': 'space-evenly','align-items': 'flex-start'}
                
                )])
        ],style={}),
        #saved data
        dcc.Store(id='crop_img'), #cropped image array (3dimensions)
        dcc.Store(id='number_subtracted'), #background subtracted number
        dcc.Store(id='array_of_img'), #downloaded array of image
        dcc.Store(id='ROI_1dimensional_data'), #1d cropped data (2dimensions, 1 list )
   
               
    ])
])

webbrowser.open_new('http://127.0.0.1:8050/') # Opens up image (on a computer server)
'''Callbacks are loops that run if a certain input has changed. They do not detect states. They usually return outputs that change
the html objects.
'''

''' This first callback is the uploading of the image. This is were we get our downloaded image.'''
@app.callback(
    Output('image_uploaded', 'src'), # image in PIL
    Output('array_of_img','data'), # image in numpy
    Output('uploaded_filename','children'), # filename
    Input('upload', 'contents'), # base64 of the downloaded image
    State('upload', 'filename'), # the name of the file 
    prevent_initial_call= True
)
def upload_image(contents,name):
    img=dash.no_update
    numArray=dash.no_update
    if contents:
        img=ImageOps.grayscale(get_pil(contents)) # converts to PIL and uses grayscale
        numArray=np.flip(np.array(ImageOps.invert(img)),1) # converts to numpy array (flip is to position it correctly)
    return img, numArray, name


'''Next is the first graphical representation. This demonstrates a 3d representation of the graph. The depth would represent the 
the ion radiation intensity.'''
 
@app.callback(
    Output('3_dimensional_graph', 'figure'),
    Input('array_of_img','data'),
    Input('crop_img','data'),
    Input('graph_histogram_colors', 'relayoutData'),
    Input('color_slider_start', 'value'),
    Input('color_slider_end', 'value'),
    prevent_initial_call= True
)
def update_3d_graph(arrayImg,cropImgData,relayout_bounds,start,end):
    Graph3d = dash.no_update
    if arrayImg:
        numArray = np.array(arrayImg)
        if cropImgData and 'crop_img.data' in dash.callback_context.triggered[0].values():
            numArray = np.array(cropImgData)        #gets the most recent data of the plot, either the upload or the crop of the ROI (this will be seen more times)
        if 'color_slider_start.value' in dash.callback_context.triggered[0].values() or 'color_slider_end.value' in dash.callback_context.triggered[0].values():
            R0 = start
            R1 = end    
        else:
            R0,R1=relayRange(relayout_bounds,numArray) # checks if the sliders or the color graph changed to change the 3d graph
        CrossZ= (np.logical_and(numArray>=R0, numArray<=R1)*numArray-R0>=0)*numArray
        Graph3d = go.Figure(data=[go.Surface(z=CrossZ, colorscale = "Greys")])
        Graph3d.update_layout(title='3D model', autosize=True, width=800, height=800, scene = dict(zaxis = dict(range=[1,255])))
        Graph3d.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True))
    return Graph3d
 


''' This graph is a supplement to the 3d graph. Using sliders and a histogram, we will be able to select a certain range of 
 intensity that will be representended on the 3d graph.'''
@app.callback(
    Output('graph_histogram_colors', 'figure'),
    Output('color_slider_start', 'value'),
    Output('color_slider_end', 'value'),
    Output('color_slider_start','max'),
    Output('color_slider_end','min'),
    Input('array_of_img','data'),
    Input('crop_img','data'),
    Input('color_slider_start', 'value'),
    Input('color_slider_end', 'value'),
    Input('graph_histogram_colors', 'relayoutData'),


    prevent_initial_call= True
)
def update_histogram_colors(arrayImg,cropImgData, start, end, slider):
    if arrayImg:
        numArray = np.array(arrayImg)
        max_start = dash.no_update
        min_end = dash.no_update
        start_value = dash.no_update
        end_value = dash.no_update
        if 'array_of_img.data' in dash.callback_context.triggered[0].values():
            max_start = np.amin(numArray)
            min_end = np.amax(numArray)
            start_value = np.amin(numArray)
            end_value = np.amax(numArray)
        if cropImgData and 'crop_img.data' in dash.callback_context.triggered[0].values():
            numArray = np.array(cropImgData)
            max_start = np.amin(numArray)
            min_end = np.amax(numArray)
            start_value = np.amin(numArray)
            end_value = np.amax(numArray)
            # code above is for the color sliders to start at the correct position and that they can not suprass eachother.
        Bin_Num = np.amax(numArray)-np.amin(numArray)
        his, bin_edges = np.histogram(numArray, bins=Bin_Num) 
        histo = go.Figure(data=[go.Scatter(x=bin_edges, y=his[1:])])
        histo.update_layout(xaxis=dict(rangeslider=dict(visible=True))) # makes and updates color histogram
        
        if 'color_slider_start.value' in dash.callback_context.triggered[0].values() or 'color_slider_end.value' in dash.callback_context.triggered[0].values() or 'graph_histogram_colors.relayoutData' in dash.callback_context.triggered[0].values():
            if 'graph_histogram_colors.relayoutData' in dash.callback_context.triggered[0].values():
                start_value,end_value = relayRange(slider,numArray)
            else:
                start_value = start
                end_value = end
            max_start = start_value 
            min_end = end_value
            histo.update_xaxes(range=(start_value,end_value)) # updates the sliders and the histogram if the other is changed

        return histo,start_value,end_value,min_end,max_start
    return dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update



''' This is the user selected ROI selection. This means that it will print the resulting graph and the standard deviation/uniformity.
To do so the user must select a given area by drag clicking and then click the crop button to crop. If error were to occur the user can click
restart to reset the image. 
'''

@app.callback(
    Output('ROI_graph','figure'),
    Output('crop_img','data'),
    Output('ROI_1dimensional_data','data'),
    Output('standard_deviation','children'),
    Output('uniformity','children'),
    Input('array_of_img','data'),
    Input('see_ROI_selection','n_clicks'),
    Input('restart_ROI_selection','n_clicks'),
    Input('crop_ROI_selection','n_clicks'),
    Input('ROI_graph', 'relayoutData'),
    Input('image_uploaded','src'),
    State('ROI_graph','figure'),
    State('number_subtracted','data'),
    State('selection_shape','value'),
    prevent_initial_call= True
    )
 
def update_ROI_and_Stdev(contents,see,restart,crop,selection,imgData,graph,sub,typeOfShape):
    
    ROI_data2Dimensional = dash.no_update
    ROI_data3Dimensional = dash.no_update
    stdev = dash.no_update
    uni = dash.no_update
    if contents:
        height, width  = np.array(contents).shape
        img = get_pil(imgData)
        if 'image_uploaded.src' in dash.callback_context.triggered[0].values() or 'restart_ROI_selection.n_clicks' in dash.callback_context.triggered[0].values():
            ROI=go.Figure()
            ROI.add_layout_image(dict(source=imgData, xref="x",yref="y",x=0,y=0, yanchor="bottom",sizing="stretch",sizex=width,sizey=height,layer = "below"))
            ROI.update_layout(title='ROI Select', autosize=True,width=900,height=900)
            ROI.update_layout(dragmode = "drawrect")  
            ROI.update_xaxes(range= (0, width),scaleanchor="y",scaleratio=1,tickwidth=1)
            ROI.update_yaxes(range= (0, height),tickwidth=1)
            ROI_data3Dimensional = None #makes ROI graph and used for reseting
        else:
            ROI = go.Figure(graph)
            

        
        if 'ROI_graph.relayoutData' in dash.callback_context.triggered[0].values() and 'shapes' in dash.callback_context.triggered[0]['value'] and  len(dash.callback_context.triggered[0]['value']['shapes']) == 1:
            ROI['layout'].update(dragmode = "pan")
        #function to make 1 rectangle
        if selection and 'see_ROI_selection.n_clicks' in dash.callback_context.triggered[0].values():
            shape = img.copy()
            draw = ImageDraw.Draw(shape)
            
            if "shapes[0].x0" in selection:
                x0,x1,y0,y1 = coords(selection["shapes[0].x0"],selection["shapes[0].x1"],height-selection["shapes[0].y0"],height-selection["shapes[0].y1"])
                if typeOfShape == 'Circle':
                    draw.ellipse((x0, y0, x1, y1), fill=None, outline=None, width=5)
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=None, outline=None, width=5) 
            elif "shapes" in selection:
                x0,x1,y0,y1 = coords(selection["shapes"][0]["x0"],selection["shapes"][0]["x1"],height-selection["shapes"][0]["y0"],height-selection["shapes"][0]["y1"])
                if typeOfShape == 'Circle':
                    draw.ellipse((x0, y0, x1, y1), fill=None, outline=None, width=5)
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=None, outline=None, width=5) 
                
            ROI['layout']['images'][0].update(source = get_base64(shape)) #converts shape into image
            # this if statement basically draws a circle/rectangle for the see function
        if selection and ("shapes[0].x0" in selection or "shapes" in selection) and 'crop_ROI_selection.n_clicks' in dash.callback_context.triggered[0].values():
            if "shapes[0].x0" in selection:
                x0,x1,y0,y1 = coords(selection["shapes[0].x0"],selection["shapes[0].x1"],height-selection["shapes[0].y0"],height-selection["shapes[0].y1"])
            elif "shapes" in selection:
                x0,x1,y0,y1 = coords(selection["shapes"][0]["x0"],selection["shapes"][0]["x1"],height-selection["shapes"][0]["y0"],height-selection["shapes"][0]["y1"])
            lum_img = Image.new('L', [width,height] , 0)
            draw = ImageDraw.Draw(lum_img)
            if typeOfShape == 'Circle':
                draw.ellipse((x0, y0, x1, y1), fill=255, outline="white")
            else:
                draw.rectangle((x0, y0, x1, y1), fill=255, outline="white")
            # code above is used for the cropping of the image
            lum_img_arr =np.array(lum_img)
            final_img_arr = np.dstack((np.array(img),lum_img_arr))
            ROI['layout']['images'][0].update(source = get_base64(Image.fromarray(final_img_arr))) #changes plot into cropped image
            ROI_data3Dimensional = makeData(img,final_img_arr)
            ROI_data2Dimensional = Check255(final_img_arr)
            #crops the NumPy array

            subtract = 0
            if sub:
                subtract = sub
            subtracted_data = [255 - i - subtract for i in ROI_data2Dimensional]
            #background subtraction
            mean_int = np.mean(subtracted_data)
            standard_dev_int = math.sqrt(np.var(subtracted_data,ddof=1))
            stdev = "Std dev: " + str(standard_dev_int)
            uni = "Uniformity: " + str((mean_int-standard_dev_int)/mean_int)
            #resultant from the crop
        return ROI,ROI_data3Dimensional,ROI_data2Dimensional, stdev, uni
 
    return dash.no_update,ROI_data3Dimensional,ROI_data2Dimensional, stdev

'''This is the background subtraction of the image. It includes the graph and the outputted number .'''

@app.callback(
    Output('background_subtration_graph','figure'),
    Output('number_subtracted','data'),
    Output('background_subtraction_output','children'),
    Input('array_of_img','data'),
    Input('calculate_background_subtraction','n_clicks'),
    State('background_subtration_graph','selectedData'),
    State('background_subtration_graph','figure'),
    prevent_initial_call= True
    )
 
def update_background_subtraction(contents,subtract,bac_range,sub_graph):
    sub = 0
    sub_out = "Background Subtracted: "
    if contents:
        numArray = np.array(contents)
        Bin_Num = np.amax(numArray)-np.amin(numArray)
        sub_data, bin_edges = np.histogram(numArray, bins=Bin_Num)
        bac_sub_graph = go.Figure()
        bac_sub_graph.add_trace(go.Scatter(x=bin_edges, y=sub_data,mode='lines',marker=dict(size=2,color='rgb(0,0,250)'),name='Histogram Values'))
        bac_sub_graph.update_layout(dragmode='select',width = 400)
        # makes the backround subtraction graph
        if bac_range and 'calculate_background_subtraction.n_clicks' in dash.callback_context.triggered[0].values():
            L,R=selectionRange(bac_range,sub_data)
            SubtractionX= np.arange(L,R)
            SubtractionY= sub_data[L-np.amin(numArray):R-np.amin(numArray)]
            Sub_Gaus_data = peakutils.peak.gaussian_fit(SubtractionX,SubtractionY,False)
            sub = Sub_Gaus_data[1]
            gaussian_y = [gaussian(x_dummy, sub, Sub_Gaus_data[2]) for x_dummy in SubtractionX]
            #uses the button and the highlighted data to get the gaussian
            trace_gaus = go.Scatter(
            x=[item_x  for item_x in SubtractionX],
            y=[item_y * Sub_Gaus_data[0] for item_y in gaussian_y],
            mode='lines',
            marker=dict(size=2,color='rgb(250,0,0)'),
            name='Gaussian Fit')
            bac_sub_graph.add_trace(trace_gaus)
            #Draws the gaussian line on the graph
            sub_out+=str(sub)
            #Returns the output


        return  bac_sub_graph,sub,sub_out
 
    return  dash.no_update, dash.no_update, dash.no_update
 
'''This is the contour graph, one of the 2 sections in the additional graphs section. In addition it displays a contour from a certain
tolerance from the median.'''

@app.callback(
    Output('contour_graph', 'figure'),
    Input('array_of_img','data'),
    Input('crop_img','data'),
    Input('tolerance_type','children'),
    Input('contour_range','value'),
    State('ROI_1dimensional_data','data'),
    prevent_initial_call= True
)
def update_Contour(arrayImg,cropImgData, ttype, tsval, OneDdata):
    Contour = dash.no_update
    if arrayImg:
        numArray = np.array(arrayImg)
        contour_stdev=np.std(numArray)
        contour_mean=np.mean(numArray) 
        if cropImgData and 'crop_img.data' in dash.callback_context.triggered[0].values():
            numArray = np.transpose(np.array(cropImgData))
            contour_stdev=np.std(OneDdata)
            contour_mean=np.mean(OneDdata) 
    #checks if its cropped or uploaded data
        if ttype == 'Sigma':
            try:
                val = float(tsval)
            except:
                val = 3 
            contour_array = np.clip(numArray,contour_mean-contour_stdev*val, contour_mean+contour_stdev*val)
        elif ttype == 'Percent':
            try:
                val = float(tsval)
            except:
                val = 15 
            contour_array = np.clip(numArray,contour_mean-contour_mean*val/100, contour_mean+contour_mean*val/100)
        #checks if Standard dev or Percent
        Contour = go.Figure(data=go.Contour(z=contour_array,colorscale='Electric'))
        Contour.update_layout(title='Contour', autosize=True, width=600, height=600)
        #returns correct graph
    return Contour

'''THis is where you select the contour tolerances, either being standard deviations from the mean or a percentage of the mean.'''
@app.callback(
    Output('tolerance_type','children'),
    Output('option_1','value'),
    Output('option_2','value'),
    Output('option_3','value'),
    Input('tolerance_type','n_clicks')
)
def change_measurement(n_clicks):
    if n_clicks%2 == 0:
        val = 'Sigma'
        op1 = '1'
        op2 = '2'
        op3 = '3'
    else:
        val = 'Percent'
        op1 = '10'
        op2 = '20'
        op3 = '30'
    return val, op1,op2,op3
    #options for the input bar (changes with standard diviation or percent button)

''' This is the X and Y graph which draws 2 lines (horizontal and vertical) on the image uploaded.
 Then we see a the insentivity that it demonstrated on the lines.'''

@app.callback(
   Output('xy_graph','figure'),
   Input('array_of_img','data'),
   Input('image_uploaded','src'),
   Input('xaxis_slider','value'),
   Input('yaxis_slider','value')
)
def update_XYgraph(contents,imgData,Xline,Yline):
    if imgData:
        height, width  = np.array(contents).shape
        XY=go.Figure()
        XY.add_layout_image(dict(source=imgData, xref="x",yref="y",x=0,y=0, yanchor="bottom",sizing="stretch",sizex=width,sizey=height,layer = "below"))
        XY.update_layout(title='XY plot')
        XY.update_xaxes(range= (0, width),showgrid=False,fixedrange = True)
        XY.update_yaxes(range= (0, height),showgrid=False,fixedrange = True)
        XY.add_vline(x=Xline)
        XY.add_hline(y=Yline)
        return XY #XY graph plotted
    else:
        return dash.no_update
    
    
'''This is the X graph of the X and y graph tab'''

@app.callback(
Output('xaxis_histogram', 'figure'),
Output('xaxis_slider','max'),
Input('array_of_img','data'),
Input('crop_img','data'),
Input('xaxis_slider','value'),
State('xaxis_slider','max'),
prevent_initial_call= True
)
def update_XHistogram(arrayImg,cropImgData,xSlider,xMax):
    Xhisto = dash.no_update
    XVal = dash.no_update
    Max= xMax+1 
    if arrayImg:
        numArray = np.array(arrayImg)
        XVal=xSlider
        if cropImgData and 'crop_img.data' in dash.callback_context.triggered[0].values():
            numArray = np.transpose(np.array(cropImgData))
        Max, width  = numArray.shape
        df = numArray[XVal]
        Xhisto = go.Figure(data= [go.Scatter(x=np.arange(Max),y=df)])
        Xhisto.update_layout(title='X Axis Histogram', autosize=True, width=1000, height=300)
        Xhisto.update_yaxes(range=[np.amin(numArray),np.amax(numArray)])
    return Xhisto,Max-1

'''This is the Y graph of the X and y graph tab'''

@app.callback(
Output('yaxis_histogram', 'figure'),
Output('yaxis_slider','max'),
Input('array_of_img','data'),
Input('crop_img','data'),
Input('yaxis_slider','value'),
State('yaxis_slider','max'),
prevent_initial_call= True
)
def update_YHistogram(arrayImg,cropImgData,ySlider,yMax):
    Yhisto = dash.no_update
    YVal = dash.no_update
    Max= yMax+1 
    if arrayImg:
        numArray = np.array(arrayImg)
        YVal=ySlider
        if cropImgData and 'crop_img.data' in dash.callback_context.triggered[0].values():
            numArray = np.transpose(np.array(cropImgData))
        length, Max  = numArray.shape
        df = numArray[YVal]
        Yhisto = go.Figure(data= [go.Scatter(x=np.arange(Max),y=df)])
        Yhisto.update_layout(title='Y Axis Histogram', autosize=True, width=1000, height=300)
        Yhisto.update_yaxes(range=[np.amin(numArray),np.amax(numArray)])
    return Yhisto,Max-1



'''This is a compliation of all the graphs to be able to print as a PDF'''


@app.callback(
    Output('uniformity_out','children'),
    Output('Stdev_out','children'),
    Output('bac_sub_out','children'),
    Output('img_out','src'),
    Output('stdev_graph_out','figure'),
    Output('3d_graph_out','figure'),
    Output("contour_graph_out","figure"),
    Output('uploaded_filename_out','children'),
    Input('tabs', 'value'),
    Input('array_of_img','data'), 
    State('uniformity','children'),
    State('standard_deviation','children'),
    State('background_subtraction_output','children'),
    State('image_uploaded', 'src'),
    State('ROI_graph','figure'),
    State('uploaded_filename','children')
    
    
    
)
def update_pdf(update,data,uniformity,standard_deviation,background_subtraction,image,standard_deviation_graph,filename):
    if update == "tab-4" and data:
        Stdev_graph = go.Figure(standard_deviation_graph)
        Stdev_graph.update_layout(height=400,width=400) #standard deviation graph
        threeD_graph = go.Figure(data=[go.Surface(z=np.array(data), colorscale = "Greys")])
        threeD_graph.update_layout(title='3D model', autosize=True, width=300, height=300, scene = dict(zaxis = dict(range=[1,255])))
        #3D graph
        numArray = np.array(data) 
        cropMean = np.mean(numArray)
        cropArray = np.clip(numArray,cropMean-cropMean*0.15, cropMean+cropMean*0.15)
        contour_graph = go.Figure(data=go.Contour(z=cropArray,colorscale='Electric'))
        contour_graph.update_layout(autosize = True, width = 300, height = 300, title='Contour Plot') 
        #contour plot is made like this because of contour plots not having some functions
        return uniformity,standard_deviation,background_subtraction,image,Stdev_graph,threeD_graph,contour_graph,filename
    return dash.no_update,dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update




'''These are all of the hand made functions I made.'''


def coords(x0,x1,y0,y1):
    return min(x0,x1),max(x0,x1),min(y0,y1),max(y0,y1)
# outputs the cords of the selection
 
def Check255(array):
    t = []
    for x in array:
        for y in x:
            if y[1] == 255:
                t.append(y[0])
    return t
# 1d ROI 
 
def makeData(initial,cropped):
    t=np.zeros(initial.size)
    xlen,ylen=initial.size
    for x in range(xlen):
        for y in range(ylen):
            if  cropped[y][x][1] == 0:
                t[x][y] = 0
            else:
                t[x][y] = cropped[y][x][0]
    return t
#2d ROI
 
def get_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/bmp;base64," + img_str.decode()
#converts arrays to images
 
def get_pil(im_b64):
    im_bytes = base64.b64decode(im_b64.split(';base64,')[1])
    im_file = BytesIO(im_bytes)  
    return(Image.open(im_file)) 
#used for uploading the images to PIL objects 
 
def relayRange(relayout_bounds,array_of_graph):
    R0=0
    R1=255
    if relayout_bounds:
        if 'xaxis.range[0]' in list(relayout_bounds.keys()):
            R0=relayout_bounds['xaxis.range[0]']
            R1=relayout_bounds['xaxis.range[1]']
        elif 'xaxis.range' in list(relayout_bounds.keys()):
            R0=relayout_bounds['xaxis.range'][0]
            R1=relayout_bounds['xaxis.range'][1]
    return R0,R1
#gets the relay range
def selectionRange(selection_bounds,data):
    if 'range' in list(selection_bounds.keys()):
        L= int(selection_bounds['range']['x'][0])
        R= int(selection_bounds['range']['x'][1])
    else:
        L=np.amin(data)
        R=np.amax(data)
    return L,R
#getst the selected range
 
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#gaussian plot
 
if __name__ == '__main__':
    app.run_server(debug=True)

