#include "glwidget.h"
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QPushButton>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QMainWindow w;
    QVBoxLayout* mainLayout = new QVBoxLayout();
    QWidget *mainWidget = new QWidget(&w);
    mainWidget->setLayout(mainLayout);
    w.setCentralWidget(mainWidget);
    mainWidget->layout()->addWidget(new QPushButton("CLIQUE"));
    mainWidget->layout()->addWidget(new GLWidget());
    w.showMaximized();
    return a.exec();
}
