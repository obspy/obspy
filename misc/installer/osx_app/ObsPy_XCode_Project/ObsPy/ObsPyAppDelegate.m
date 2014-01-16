//
//  ObsPyAppDelegate.m
//  ObsPy
//
//  Created by Lion Krischer on 27.8.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "ObsPyAppDelegate.h"

@implementation ObsPyAppDelegate

@synthesize window;
@synthesize dataModel;
@synthesize terminalEmulationWindowCloseButton;
@synthesize terminalEmulationWindowStopButton;
@synthesize terminalRunning;


- (id)init
{
    self = [super init];
    if (self) {
        // Init the data model.
        dataModel = [[ApplicationDataModel alloc] init];
        [self setObsPyHyperlinkWithTextField:ObsPyLink];
        
        // Handle the scipy symlinking every time the application starts.
        //`[self setSciPySymlink];

    }
    
    return self;
}


- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    // Insert code here to initialize your application
    NSString*	version = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleVersion"];
    [dataModel setApplicationVersion:version];
    
    [self setObsPyHyperlinkWithTextField:ObsPyLink];
    
    
}

- (IBAction)launchIpython:(id)pId;
{
    
    [[NSWorkspace sharedWorkspace] openFile:@"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/ipython" withApplication:@"Terminal"];

}

- (IBAction)launchIpythonNotebook:(id)pId;
{
    
    [[NSWorkspace sharedWorkspace] openFile:@"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/launch_ipython_notebook" withApplication:@"Terminal"];
    
}


- (IBAction)runQuickTest:(id)pId;
{
    NSString * path = [[NSBundle mainBundle] pathForResource:@"quick_test" ofType:@"py"];
    emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
    [emulator runSingleCommand: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/python" withArguments:[NSArray arrayWithObjects:path, nil]];
    
}

- (IBAction)runObsPyTests:(id)pId;
{
    // Show alert sheet and give the option to report the results.
    NSAlert *alert = [NSAlert alertWithMessageText:@"Report the results of the ObsPy tests?"
                      defaultButton:@"Yes" alternateButton:@"Cancel" otherButton:@"No"
                      informativeTextWithFormat:@"Results will be posted at http://tests.obspy.org"];
    [alert beginSheetModalForWindow:window modalDelegate:self didEndSelector:@selector(obspyRuntestAlertDidEnd:returnCode:contextInfo:) contextInfo:nil];
}


- (void)obspyRuntestAlertDidEnd:(NSAlert *)alert returnCode:(NSInteger)retcode contextInfo:(id)contextInfo
{
    [[alert window] close];
    // Yes button pressed.
    if (retcode == NSAlertDefaultReturn)
    {
        emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
        [emulator runSingleCommand: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/obspy-runtests" withArguments:[NSArray arrayWithObjects:@"-r", nil]];
    }
    
    // Cancel button pressed.
    if (retcode == NSAlertAlternateReturn)
    {
        return;
    }
    
    // No button pressed.
    if (retcode == NSAlertOtherReturn)
    {
        emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
        [emulator runSingleCommand: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/obspy-runtests" withArguments:[NSArray arrayWithObjects:@"-d", nil]];
    }
}


- (IBAction)runNumpyTests:(id)pId;
{
    NSString * path = [[NSBundle mainBundle] pathForResource:@"numpy_tests" ofType:@"py"];  
    emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
    [emulator runSingleCommand: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/python" withArguments:[NSArray arrayWithObjects:path, nil]];
    
}


- (IBAction)runScipyTests:(id)pId;
{
    NSString * path = [[NSBundle mainBundle] pathForResource:@"scipy_tests" ofType:@"py"];
    emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
    [emulator runSingleCommand: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/python" withArguments:[NSArray arrayWithObjects:path, nil]];
    
}


- (IBAction)copyEmulatorOutputToClipboard:(id)sender {
    NSPasteboard *pasteBoard = [NSPasteboard generalPasteboard];
    [pasteBoard declareTypes:[NSArray arrayWithObjects:NSStringPboardType, nil] owner:nil];
    [pasteBoard setString: [OutPut string] forType:NSStringPboardType];
}


- (IBAction)stopCurrentEmulatorTask:(id)sender
{
    [emulator stopTask];
}

- (IBAction)orderEmulatorWindowOut:(id)sender
{
    [NSApp endSheet:terminalWindow returnCode:NSOKButton];
    [terminalWindow orderOut:nil];
}


//- (IBAction)showScipyPath:(id)sender
//{
//    NSString * scipy_path;
//    scipy_path = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:@"/Applications/ObsPy.app/Contents/MacOS/lib/python2.7/site-packages/scipy" error:nil];
//
//    
//    // Show alert sheet and give the option to report the results.
//    NSAlert *alert = [NSAlert alertWithMessageText:@"Path of the SciPy version currently used:"
//                                     defaultButton:@"OK" alternateButton:nil otherButton:nil
//                         informativeTextWithFormat:scipy_path];
//    [alert beginSheetModalForWindow:window modalDelegate:self didEndSelector:@selector(scipyPathAlertDidEnd:returnCode:contextInfo:) contextInfo:nil];
//}


//// Sets a scipy symlink according to the current OSX version.
//-(void)setSciPySymlink
//{
//    NSString * symlink_path;
//    NSString * osx_lion_path;
//    NSString * snow_leopard_path;
//    
//    symlink_path = @"/Applications/ObsPy.app/Contents/MacOS/lib/python2.7/site-packages/scipy";
//    osx_lion_path = @"/Applications/ObsPy.app/Contents/MacOS/extras/scipy_osx_lion";
//    snow_leopard_path = @"/Applications/ObsPy.app/Contents/MacOS/extras/scipy_osx_snow_leopard";
//    
//    NSFileManager * file_manager = [NSFileManager defaultManager];
//    
//    // Check if the symlink already exists and delete it in case it does.
//    if ([file_manager fileExistsAtPath:symlink_path])
//    {
//        [file_manager removeItemAtPath:symlink_path error:nil];
//    }
//    
//    // Get the minor version number.
//    int version = 0;
//    Gestalt(gestaltSystemVersionMinor, &version);
//    // Set to lion if larger or equal to seven.
//    if (version >= 7)
//    {
//        [file_manager createSymbolicLinkAtPath:symlink_path withDestinationPath:osx_lion_path error:nil];
//    }
//    else
//    {
//        [file_manager createSymbolicLinkAtPath:symlink_path withDestinationPath:snow_leopard_path error:nil];
//    }
//}


//- (void)scipyPathAlertDidEnd:(NSAlert *)alert returnCode:(NSInteger)retcode contextInfo:(id)contextInfo
//{
//    [[alert window] close];
//}


- (IBAction)startVirtualEnvAssistant:(id)sender
{
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    [panel setDirectoryURL: [dataModel virtualEnvPath]];
    [panel setCanChooseFiles:NO];
    [panel setCanChooseDirectories:YES];
    [panel setMessage:@"Choose the (empty) directory where you want the virtual enviroment to be created in."];
    [panel setPrompt:@"Next"];
    [panel setCanCreateDirectories:YES];
    [panel beginSheetModalForWindow:window completionHandler:^(NSInteger returnCode)
    {
        if (returnCode == NSOKButton)
        {
            [NSApp endSheet:panel returnCode:NSOKButton];
            [panel orderOut:nil];
            [self launchVirtualEnvironmentAssistantSettings:[[panel URLs] objectAtIndex:0]];
            //[self createVirtualEnvAtDirectory: [[panel URLs] objectAtIndex:0]];
        }
        else {
            [panel orderOut:nil];
            [NSApp endSheet:panel returnCode:NSCancelButton];
        }
    }];

}

-(IBAction)cancelAssistant:(id)sender
{
    [NSApp endSheet:virtualEnvAssistantWindow returnCode:NSCancelButton];
    [virtualEnvAssistantWindow orderOut:nil];
}

- (void)launchVirtualEnvironmentAssistantSettings: (NSURL *)url
{
    [dataModel setVirtualEnvPath:url];
    [NSApp beginSheet:virtualEnvAssistantWindow modalForWindow:window modalDelegate:self didEndSelector:NULL contextInfo:nil];
}

- (IBAction)reselectVirtualEnvDir:(id)sender
{
    [NSApp endSheet:virtualEnvAssistantWindow returnCode:NSCancelButton];
    [virtualEnvAssistantWindow orderOut:nil];
    [self startVirtualEnvAssistant:self];
}

- (IBAction)createVirtualEnv:(id)sender {
    // End old sheed.
    [NSApp endSheet:virtualEnvAssistantWindow returnCode:NSCancelButton];
    [virtualEnvAssistantWindow orderOut:nil];
    
    emulator = [[TTYEmulation alloc] initWithTextView:OutPut andViewWindow:terminalWindow];
    
    NSMutableArray * cmdArray;
    
    NSString * promptName;
    promptName = [dataModel promptName];
    
    cmdArray = [[NSMutableArray alloc] init];
    // Create the commands to be executed.
    NSArray * cmd1 = [[NSArray alloc] initWithObjects: @"/Applications/ObsPy.app/Contents/MacOS/Python.framework/Versions/2.7/bin/virtualenv", [[NSArray alloc] initWithObjects:@"--distribute", @"--unzip-setuptools", [NSString stringWithFormat: @"--prompt=%@", promptName], [[dataModel virtualEnvPath] path], nil], nil];
    [cmdArray addObject:cmd1];
    
    if ([[dataModel virtualEnvInstallReadline] boolValue]) {
        NSArray * cmd2 = [[NSArray alloc] initWithObjects:[[[[dataModel virtualEnvPath] URLByAppendingPathComponent:@"bin"] URLByAppendingPathComponent:@"easy_install"] path], [[NSArray alloc] initWithObjects: @"readline", nil], nil];
        [cmdArray addObject:cmd2];
    }
    
    if ([[dataModel virtualEnvInstallIPython] boolValue]) {
        NSArray * cmd3 = [[NSArray alloc] initWithObjects:[[[[dataModel virtualEnvPath] URLByAppendingPathComponent:@"bin"] URLByAppendingPathComponent:@"easy_install"] path], [[NSArray alloc] initWithObjects:@"-U", @"ipython", nil], nil];
        [cmdArray addObject:cmd3];
    }
    
    if ([[dataModel virtualEnvUpgradeDistribute] boolValue]) {
        NSArray * cmd4 = [[NSArray alloc] initWithObjects:[[[[dataModel virtualEnvPath] URLByAppendingPathComponent:@"bin"] URLByAppendingPathComponent:@"pip"] path], [[NSArray alloc] initWithObjects:@"install", @"--upgrade", @"distribute", nil], nil];
        [cmdArray addObject:cmd4];
    }
    
    // Runt the commands.
    [emulator runMultipleCommands: [[NSArray alloc] initWithArray:cmdArray]];
    
    if ([[dataModel virtualEnvAddToBashProfile] boolValue]) {
        [self appendToBashProfile];
    }

}

-(void)appendToBashProfile {
    NSString * homePath;
    homePath = @"~";
    NSURL * home;
    home = [[NSURL alloc] initFileURLWithPath:[homePath stringByExpandingTildeInPath] isDirectory:YES];
    NSString * cmd_string = [NSString stringWithFormat: @"source %@\n", [[[[dataModel virtualEnvPath] URLByAppendingPathComponent:@"bin"] URLByAppendingPathComponent:@"activate"] path]];
    
    [[NSFileManager defaultManager] removeItemAtURL:[home URLByAppendingPathComponent:@".bash_profile_obspy_bak" isDirectory:NO] error:nil];
    
    [[NSFileManager defaultManager] copyItemAtURL:[home URLByAppendingPathComponent:@".bash_profile" isDirectory:NO] toURL:[home URLByAppendingPathComponent:@".bash_profile_obspy_bak" isDirectory:NO] error:nil];
    
    // Append to file and close it.
    NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:[[home URLByAppendingPathComponent:@".bash_profile" isDirectory:NO] path]];
    [fileHandle seekToEndOfFile];
    [fileHandle writeData:[cmd_string dataUsingEncoding:NSUTF8StringEncoding]];
    [fileHandle closeFile];

}


// Modified copy from from: http://developer.apple.com/library/mac/#qa/qa1487/_index.html
-(void)setObsPyHyperlinkWithTextField:(NSTextField*)inTextField
{
    // both are needed, otherwise hyperlink won't accept mousedown
    [inTextField setAllowsEditingTextAttributes: YES];
    [inTextField setSelectable: YES];
    
    NSURL* url = [NSURL URLWithString:@"http://obspy.org"];
    
    NSMutableAttributedString* string = [[NSMutableAttributedString alloc] init];
    [string appendAttributedString: [NSAttributedString hyperlinkFromString:@"http://obspy.org" withURL:url]];
    
    // set the attributed string to the NSTextField
    [inTextField setAttributedStringValue: string];
}

-(BOOL) applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication
{
    return YES;
}

@end
