//
//  ObsPyAppDelegate.h
//  ObsPy
//
//  Created by Lion Krischer on 27.8.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "ApplicationDataModel.h"
#import "Hyperlink.h"
#import "TTYEmulation.h"

@interface ObsPyAppDelegate : NSObject <NSApplicationDelegate> {
    NSWindow *window;
    ApplicationDataModel * dataModel;
    
    TTYEmulation * emulator;
    
    IBOutlet NSTextField * ObsPyLink;
    IBOutlet NSTextView * OutPut;
    IBOutlet NSWindow * virtualEnvAssistantWindow;
    IBOutlet NSWindow * terminalWindow;
    IBOutlet NSButton * terminalEmulationWindowCloseButton;
    IBOutlet NSButton * terminalEmulationWindowStopButton;
    IBOutlet NSProgressIndicator * terminalRunning;
}

@property (assign) IBOutlet NSWindow *window;

@property (assign) IBOutlet NSButton * terminalEmulationWindowCloseButton;
@property (assign) IBOutlet NSButton * terminalEmulationWindowStopButton;
@property (assign) IBOutlet NSProgressIndicator * terminalRunning;

@property (nonatomic, retain) ApplicationDataModel * dataModel;

- (IBAction)launchIpython:(id)pId;
- (IBAction)launchIpythonNotebook:(id)pId;
- (IBAction)runQuickTest:(id)pId;
- (IBAction)runObsPyTests:(id)pId;
- (IBAction)runNumpyTests:(id)pId;
- (IBAction)runScipyTests:(id)pId;
//- (IBAction)showScipyPath:(id)pId;


- (IBAction)copyEmulatorOutputToClipboard:(id)sender;
- (IBAction)stopCurrentEmulatorTask:(id)sender;
- (IBAction)orderEmulatorWindowOut:(id)sender;
- (IBAction)startVirtualEnvAssistant:(id)sender;
- (IBAction)createVirtualEnv:(id)sender;
- (IBAction)reselectVirtualEnvDir:(id)sender;
- (IBAction)cancelAssistant:(id)sender;
- (void)launchVirtualEnvironmentAssistantSettings: (NSURL *)url;
- (void)appendToBashProfile;

-(void)setObsPyHyperlinkWithTextField:(NSTextField*)inTextField;

//-(void)setSciPySymlink;

- (void)obspyRuntestAlertDidEnd:(id)alert returnCode:(NSInteger)retcode contextInfo:(id)contextInfo;

@end
