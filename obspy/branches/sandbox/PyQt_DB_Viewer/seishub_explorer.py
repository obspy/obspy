from PyQt4 import QtCore, QtGui, QtWebKit

from gui import MainWindow, Environment, Website, Timers
import os
import sys


class TabWidget(QtGui.QTabWidget):
    """
    Very basic class that handles the tabs.
    """
    def __init__(self, parent=None):
       super (TabWidget, self).__init__(parent)

if __name__ == '__main__':
    # Init the environment.
    env = Environment(debug = True)
    # Init QApplication.
    env.qApp = QtGui.QApplication(sys.argv)

    # Removes the frame around the widgets in the status bar. May apparently
    # cause some other problems:
    # http://www.qtcentre.org/archive/index.php/t-1904.html
    env.qApp.setStyleSheet("QStatusBar::item {border: 0px solid black}")

    # Init splash screen to show something is going on.
    pixmap = QtGui.QPixmap(os.path.join(env.res_dir, 'splash.png'))
    env.splash = QtGui.QSplashScreen(pixmap)
    env.splash.show()
    env.splash.showMessage('Init interface...', QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
                   QtCore.Qt.black) 
    # Force Qt to uodate outside of any event loop to display the splash
    # screen.
    env.qApp.processEvents()

    # Init the main application window.
    window = QtGui.QMainWindow()

    # Init Tabbed Interface.
    tab = TabWidget()

    # Init the main tab and add it to the tabs-manager.
    main_window = MainWindow(env = env)
    tab.addTab(main_window, 'Main View')

    # Create the map tab that just displays a web page.
    env.web = QtWebKit.QWebView()
    env.web.show()
    tab.addTab(env.web, 'Map')

    # Add the station browser tab.
    env.station_browser = QtWebKit.QWebView()
    env.station_browser.show()
    tab.addTab(env.station_browser, 'Station Browser')
    
    # Init Status bar.
    st = QtGui.QStatusBar()
    env.st = st

    # Label to display the server status in the status bar. Need to init with
    # rich text if rich text should be used later on.
    env.server_status = QtGui.QLabel('<font color="#FF0000"></font>')
    st.addPermanentWidget(env.server_status)
    # Seperator Label.
    env.seperator = QtGui.QLabel()
    # XXX: Not working with PyQt 4.4.
    #env.seperator.setFrameShape(5)
    env.seperator.setFrameShadow(QtGui.QFrame.Raised)
    st.addPermanentWidget(env.seperator)
    # Label to display the current UTC time.
    env.current_time = QtGui.QLabel('')
    st.addPermanentWidget(env.current_time)

    # Start the timers.
    timers = Timers(env = env)

    window.setCentralWidget(tab)
    # Set the status bar.
    window.setStatusBar(st)
    window.resize(1150, 700)
    # Startup the rest.
    main_window.startup()
    env.qApp.processEvents()

    # After everything is loaded show the main window and close the splash
    # screen.
    window.show()
    env.splash.finish(window)
    # Some graphics can only be drawn once the window is showing.
    main_window.graphics_start()
    env.qApp.exec_()
