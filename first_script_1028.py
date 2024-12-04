# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Version 2022.1.0
# 19:13:20  10月 28, 2024
# ----------------------------------------------
import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oProject.SaveAs("C:\\Users\\24762\\Desktop\\Graduation Project_lzk\\patch_by_python.aedt", True)
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oDesign.SetSolutionType("HFSS Modal Network", 
	[
		"NAME:Options",
		"EnableAutoOpen:="	, False
	])
oDesign.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:LocalVariableTab",
			[
				"NAME:PropServers", 
				"LocalVariables"
			],
			[
				"NAME:NewProps",
				[
					"NAME:H",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "1.6mm"
				],
				[
					"NAME:L0",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "30.21mm"
				],
				[
					"NAME:W0",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "37.26mm"
				],
				[
					"NAME:L1",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "17.45mm"
				],
				[
					"NAME:W1",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "1.16mm"
				],
				[
					"NAME:L2",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "15mm"
				],
				[
					"NAME:W2",
					"PropType:="		, "VariableProp",
					"UserDef:="		, True,
					"Value:="		, "2.98mm"
				]
			]
		]
	])
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateBox(
	[
		"NAME:BoxParameters",
		"XPosition:="		, "-0.5mm",
		"YPosition:="		, "-0.5mm",
		"ZPosition:="		, "0mm",
		"XSize:="		, "1mm",
		"YSize:="		, "1mm",
		"ZSize:="		, "0.1mm"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Box1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Box1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Name",
					"Value:="		, "substrate"
				],
				[
					"NAME:Material",
					"Value:="		, "\"FR4_epoxy\""
				],
				[
					"NAME:Color",
					"R:="			, 143,
					"G:="			, 143,
					"B:="			, 175
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.7
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"substrate:CreateBox:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Position",
					"X:="			, "-L0",
					"Y:="			, "-W0",
					"Z:="			, "0mm"
				],
				[
					"NAME:XSize",
					"Value:="		, "1.5*L0+L1+L2"
				],
				[
					"NAME:YSize",
					"Value:="		, "2*W0"
				],
				[
					"NAME:ZSize",
					"Value:="		, "H"
				]
			]
		]
	])
oEditor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "-10mm",
		"YStart:="		, "-15mm",
		"ZStart:="		, "0mm",
		"Width:="		, "25mm",
		"Height:="		, "30mm",
		"WhichAxis:="		, "Z"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Rectangle1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Rectangle1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Material Appearance",
					"Value:="		, True
				],
				[
					"NAME:Display Wireframe",
					"Value:="		, True
				],
				[
					"NAME:Name",
					"Value:="		, "patch"
				],
				[
					"NAME:Color",
					"R:="			, 128,
					"G:="			, 128,
					"B:="			, 64
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.7
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"patch:CreateRectangle:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Position",
					"X:="			, "-L0/2",
					"Y:="			, "-W0/2",
					"Z:="			, "H"
				],
				[
					"NAME:XSize",
					"Value:="		, "L0"
				],
				[
					"NAME:YSize",
					"Value:="		, "W0"
				]
			]
		]
	])
oEditor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "16mm",
		"YStart:="		, "-4mm",
		"ZStart:="		, "0mm",
		"Width:="		, "14mm",
		"Height:="		, "8mm",
		"WhichAxis:="		, "Z"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Rectangle1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Rectangle1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Name",
					"Value:="		, "TLINE"
				],
				[
					"NAME:Color",
					"R:="			, 128,
					"G:="			, 128,
					"B:="			, 64
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.7
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"TLINE:CreateRectangle:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Position",
					"X:="			, "L0/2",
					"Y:="			, "-W1/2",
					"Z:="			, "H"
				],
				[
					"NAME:XSize",
					"Value:="		, "L1"
				],
				[
					"NAME:YSize",
					"Value:="		, "W1"
				]
			]
		]
	])
oEditor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "40mm",
		"YStart:="		, "4mm",
		"ZStart:="		, "0mm",
		"Width:="		, "-8mm",
		"Height:="		, "-8mm",
		"WhichAxis:="		, "Z"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Rectangle1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Rectangle1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Name",
					"Value:="		, "50ohm"
				],
				[
					"NAME:Color",
					"R:="			, 128,
					"G:="			, 128,
					"B:="			, 64
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.7
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"50ohm:CreateRectangle:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Position",
					"X:="			, "L0/2+L1",
					"Y:="			, "-W2/2",
					"Z:="			, "H"
				],
				[
					"NAME:XSize",
					"Value:="		, "L2"
				],
				[
					"NAME:YSize",
					"Value:="		, "W2"
				]
			]
		]
	])
oEditor.Unite(
	[
		"NAME:Selections",
		"Selections:="		, "50ohm,patch,TLINE"
	], 
	[
		"NAME:UniteParameters",
		"KeepOriginals:="	, False
	])
oModule = oDesign.GetModule("BoundarySetup")
oModule.AssignPerfectE(
	[
		"NAME:PerfE1",
		"Objects:="		, ["50ohm"],
		"InfGroundPlane:="	, False
	])
oEditor.CreateBox(
	[
		"NAME:BoxParameters",
		"XPosition:="		, "-40mm",
		"YPosition:="		, "35mm",
		"ZPosition:="		, "0mm",
		"XSize:="		, "110mm",
		"YSize:="		, "-75mm",
		"ZSize:="		, "30mm"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Box1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Box1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Name",
					"Value:="		, "airBox"
				],
				[
					"NAME:Color",
					"R:="			, 128,
					"G:="			, 255,
					"B:="			, 128
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.8
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"airBox:CreateBox:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Position",
					"X:="			, "-L0/2-30mm",
					"Y:="			, "-W0/2-30mm",
					"Z:="			, "0mm"
				],
				[
					"NAME:XSize",
					"Value:="		, "L0+L1+L2+30mm"
				],
				[
					"NAME:YSize",
					"Value:="		, "W0+60mm"
				],
				[
					"NAME:ZSize",
					"Value:="		, "H+30mm"
				]
			]
		]
	])
oModule.AssignRadiation(
	[
		"NAME:Rad1",
		"Objects:="		, ["airBox"]
	])
oModule.AssignPerfectE(
	[
		"NAME:PerfE2",
		"Faces:="		, [8],
		"InfGroundPlane:="	, False
	])
oEditor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "47.555mm",
		"YStart:="		, "-18.63mm",
		"ZStart:="		, "0mm",
		"Width:="		, "33.63mm",
		"Height:="		, "10mm",
		"WhichAxis:="		, "X"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Rectangle1",
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"ShellElement:="	, False,
		"ShellElementThickness:=", "0mm",
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				"Rectangle1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Name",
					"Value:="		, "port1"
				],
				[
					"NAME:Color",
					"R:="			, 255,
					"G:="			, 128,
					"B:="			, 128
				],
				[
					"NAME:Transparent",
					"Value:="		, 0.5
				]
			]
		]
	])
oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DCmdTab",
			[
				"NAME:PropServers", 
				"port1:CreateRectangle:1"
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:YSize",
					"Value:="		, "8*W2"
				],
				[
					"NAME:ZSize",
					"Value:="		, "8*H"
				],
				[
					"NAME:Position",
					"X:="			, "L0/2+L1+L2",
					"Y:="			, "-4*W2",
					"Z:="			, "0mm"
				]
			]
		]
	])
oModule.AssignWavePort(
	[
		"NAME:1",
		"Faces:="		, [113],
		"NumModes:="		, 1,
		"UseLineModeAlignment:=", False,
		"DoDeembed:="		, False,
		"RenormalizeAllTerminals:=", True,
		[
			"NAME:Modes",
			[
				"NAME:Mode1",
				"ModeNum:="		, 1,
				"UseIntLine:="		, False,
				"CharImp:="		, "Zpi",
				"RenormImp:="		, "50ohm"
			]
		],
		"ShowReporterFilter:="	, False,
		"ReporterFilter:="	, [True],
		"UseAnalyticAlignment:=", False
	])
oModule = oDesign.GetModule("AnalysisSetup")
oModule.InsertSetup("HfssDriven", 
	[
		"NAME:Setup1",
		"SolveType:="		, "Single",
		"Frequency:="		, "2.45GHz",
		"MaxDeltaS:="		, 0.02,
		"UseMatrixConv:="	, False,
		"MaximumPasses:="	, 20,
		"MinimumPasses:="	, 1,
		"MinimumConvergedPasses:=", 1,
		"PercentRefinement:="	, 30,
		"IsEnabled:="		, True,
		[
			"NAME:MeshLink",
			"ImportMesh:="		, False
		],
		"BasisOrder:="		, 1,
		"DoLambdaRefine:="	, True,
		"DoMaterialLambda:="	, True,
		"SetLambdaTarget:="	, False,
		"Target:="		, 0.3333,
		"UseMaxTetIncrease:="	, False,
		"PortAccuracy:="	, 2,
		"UseABCOnPort:="	, False,
		"SetPortMinMaxTri:="	, False,
		"DrivenSolverType:="	, "Direct Solver",
		"EnhancedLowFreqAccuracy:=", False,
		"SaveRadFieldsOnly:="	, False,
		"SaveAnyFields:="	, True,
		"IESolverType:="	, "Auto",
		"LambdaTargetForIESolver:=", 0.15,
		"UseDefaultLambdaTgtForIESolver:=", True,
		"IE Solver Accuracy:="	, "Balanced",
		"InfiniteSphereSetup:="	, ""
	])
oModule.InsertFrequencySweep("Setup1", 
	[
		"NAME:Sweep",
		"IsEnabled:="		, True,
		"RangeType:="		, "LinearStep",
		"RangeStart:="		, "1.5GHz",
		"RangeEnd:="		, "3.5GHz",
		"RangeStep:="		, "0.01GHz",
		"Type:="		, "Interpolating",
		"SaveFields:="		, False,
		"SaveRadFields:="	, False,
		"InterpTolerance:="	, 0.5,
		"InterpMaxSolns:="	, 250,
		"InterpMinSolns:="	, 0,
		"InterpMinSubranges:="	, 1,
		"InterpUseS:="		, True,
		"InterpUsePortImped:="	, False,
		"InterpUsePropConst:="	, True,
		"UseDerivativeConvergence:=", False,
		"InterpDerivTolerance:=", 0.2,
		"UseFullBasis:="	, True,
		"EnforcePassivity:="	, True,
		"PassivityErrorTolerance:=", 0.0001,
		"SMatrixOnlySolveMode:=", "Auto"
	])
oProject.Save()
oDesign.AnalyzeAll()
oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport("S Parameter Plot 1", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep", 
	[
		"Domain:="		, "Sweep"
	], 
	[
		"Freq:="		, ["All"],
		"H:="			, ["Nominal"],
		"L0:="			, ["Nominal"],
		"W0:="			, ["Nominal"],
		"L1:="			, ["Nominal"],
		"W1:="			, ["Nominal"],
		"L2:="			, ["Nominal"],
		"W2:="			, ["Nominal"]
	], 
	[
		"X Component:="		, "Freq",
		"Y Component:="		, ["dB(S(1,1))"]
	])
oProject.Save()
oModule.ExportToFile("S Parameter Plot 1", "C:/Users/24762/Desktop/Graduation Project_lzk/S Parameter Plot 1.csv", False)
