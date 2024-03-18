#ifndef SYSTEM_3D_GLOBAL_H
#define SYSTEM_3D_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(SYSTEM_3D_LIBRARY)
#define SYSTEM_3D_EXPORT Q_DECL_EXPORT
#else
#define SYSTEM_3D_EXPORT Q_DECL_IMPORT
#endif

#endif // SYSTEM_3D_GLOBAL_H
