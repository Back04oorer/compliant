#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <X11/Xlib.h>
#include <xcb/xcbext.h>

/* X11 and the container have separate IPC namespaces; use ordinary image copies. */
Bool XQueryExtension(Display *display, const char *name, int *opcode, int *event, int *error)
{
    if (name && strcmp(name, "MIT-SHM") == 0)
        return False;
    typedef Bool (*query_fn)(Display *, const char *, int *, int *, int *);
    query_fn query = (query_fn)dlsym(RTLD_NEXT, "XQueryExtension");
    return query ? query(display, name, opcode, event, error) : False;
}

const xcb_query_extension_reply_t *xcb_get_extension_data(xcb_connection_t *connection,
                                                        xcb_extension_t *extension)
{
    static const xcb_query_extension_reply_t absent = {.response_type = 1};
    if (extension && extension->name && strcmp(extension->name, "MIT-SHM") == 0)
        return &absent;
    typedef const xcb_query_extension_reply_t *(*query_fn)(xcb_connection_t *, xcb_extension_t *);
    query_fn query = (query_fn)dlsym(RTLD_NEXT, "xcb_get_extension_data");
    return query ? query(connection, extension) : &absent;
}
