//
//  Hyperlink.h
//
// Copied straight from: http://developer.apple.com/library/mac/#qa/qa1487/_index.html

#import <Foundation/Foundation.h>

@interface NSAttributedString (Hyperlink)
+(id)hyperlinkFromString:(NSString*)inString withURL:(NSURL*)aURL;
@end
