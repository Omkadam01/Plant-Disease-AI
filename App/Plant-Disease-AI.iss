#define MyAppName "Plant Disease AI"
#define MyAppVersion "1.0"
#define MyAppExeName "Plant-Disease-AI.exe"

[Setup]
AppId={{CBAF67F7-38C5-4F66-9DC4-0B61A54C7DF7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\Plant-Disease-AI
DefaultGroupName={#MyAppName}
OutputBaseFilename=Plant-Disease-AI-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\App\{#MyAppExeName}

[Dirs]
Name: "{app}\Outputs"

[Files]
Source: "{#SourcePath}\{#MyAppExeName}"; DestDir: "{app}\App"; Flags: ignoreversion
Source: "{#SourcePath}\..\Models\Final_Model\*"; DestDir: "{app}\Models\Final_Model"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourcePath}\..\Dataset\datasets\grape\reports\class_mapping.json"; DestDir: "{app}\Dataset\datasets\grape\reports"; Flags: ignoreversion
Source: "{#SourcePath}\..\Dataset\datasets\tomato\reports\class_mapping.json"; DestDir: "{app}\Dataset\datasets\tomato\reports"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\App\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\App\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\App\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
