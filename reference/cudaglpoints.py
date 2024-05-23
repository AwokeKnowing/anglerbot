import sys

from cuda import cudart

import numpy as np
import cupy as cp

import pyrr
import glfw

from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders



def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)

        self._graphics_ressource = None
        self._cuda_buffer = None

        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

        return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array

    




VERTEX_SHADER = """
#version 330

in vec3 position;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(position, 1.0f);
}
"""


FRAGMENT_SHADER = """
#version 330

out vec4 outColor;

void main() {
    outColor = vec4(0.0f, 0.7f, 0.0f, 1.0f);
}
"""

class PointCloudGL:
    def __init__(self, max_vertices=1000000, point_size=2.0):
        self.max_vertices = max_vertices

        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.positionLoc = glGetAttribLocation(self.shader, "position")
        self.transformLoc = glGetUniformLocation(self.shader, "transform")

        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)
        glPointSize(point_size)

        self.cuda_vertex_buffer = self.setup_buffers(self.max_vertices)

    def setup_buffers(self, max_vertices):

        ftype = np.float32

        vertex_bytes = 3 * max_vertices * ftype().nbytes
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
        vertex_buffer = CudaOpenGLMappedArray(ftype, (max_vertices, 3), VBO, flags)

        return vertex_buffer
    
    def draw(self, num_points, width, height):
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        rot_x = pyrr.Matrix44.from_x_rotation(0)
        rot_y = pyrr.Matrix44.from_y_rotation(0)

        glUniformMatrix4fv(self.transformLoc, 1, GL_FALSE, rot_x * rot_y)

        glBindBuffer(GL_ARRAY_BUFFER, self.cuda_vertex_buffer.gl_buffer)

        glEnableVertexAttribArray(self.positionLoc)
        glVertexAttribPointer(self.positionLoc, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glDrawArrays(GL_POINTS, 0, num_points)

    def __del__(self):
        self.cuda_vertex_buffer.unregister()

    

class WindowsGLFW:
    def __init__(self, width=640, height=480, title="GL Window", aspect_w=None, aspect_h=None, fps=False):
        if not glfw.init(): 
            return
        
        self.fps = fps
        self.winCount = 0
        self.fps_stats = {}
        self.windows = {}

        win = self.create_window('main',width, height, title, aspect_w, aspect_h)
               
        #shortcut to main glfw window
        self.active = 'main'
        self.win = win
        self.size = (width, height)
        self.width = width
        self.height = height
        

    def title(self,title=None,win_name=None,store=True):
        if win_name is None:
            win_name = self.active

        window = self.windows[win_name]['win']

        if title is None:
            return glfw.get_window_title(window)
        else:
            glfw.set_window_title(window,title)
            if store:
                self.windows[win_name]['title'] = title


    def create_window(self, name,width=640, height=480, title="GL Window", aspect_w=None,aspect_h=None):
        if aspect_w is None: 
            aspect_w = width
        if aspect_h is None:
            aspect_h = height
        
        window = glfw.create_window(width, height, title, None, None)
        self.winCount += 1
        glfw.set_window_user_pointer(window, self.winCount) # used to compare. eg see onSize below
        glfw.set_window_aspect_ratio(window, aspect_w,aspect_h)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        
        self.fps_stats[name] = {'last_time':self.time(),'frames':0}

        def onSize(window,w,h):
            obj = glfw.get_window_user_pointer(window)
            for k, v in self.windows.items():
                obj2=glfw.get_window_user_pointer(v['win'])
                if not obj is obj2:
                    continue

                v['size']=(w,h)
                if self.win is v['win']:
                    self.size=(w,h)
                    self.width = w
                    self.height = h
                break

        glfw.set_window_size_callback(window, onSize)

        self.windows[name] = {'size':(width, height), 'win': window,'title':title, 'fps':60}

        return window
    
    
    def closed(self, win_name=None):
        if win_name is None:
            return glfw.window_should_close(self.win)
        else:
            return glfw.window_should_close(self.windows[win_name]['win'])
        
    
    def time(self):
        return glfw.get_time()
    
    
    def activate(self,win_name):
        w = self.windows[win_name]

        self.active = win_name
        self.win = w.win
        self.size=self.size = w.size 
        self.width,self.height = self.size 


    def update_fps(self):
        stat = self.fps_stats[self.active]
        t = glfw.get_time()
        dt = t - stat['last_time']
        stat['frames'] += 1
        if dt >= 1.0:
            win = self.windows[self.active]    
            win['fps'] = stat['frames'] / dt
            stat['last_time'] = t
            stat['frames'] = 0
            self.title(f"{win['title']} ({win['fps']:.1f} fps)", store=False)

    
    def swap(self):
        if self.fps:
            self.update_fps()

        glfw.swap_buffers(self.win)
        glfw.poll_events()

    
    def __del__(self):
        try:
            glfw.terminate()
        except TypeError:
            pass


def main():  
    gl = WindowsGLFW(800,800, "CuPy Cuda/OpenGL Map", 1,1,fps=True)
    pcv = PointCloudGL(848*480)

    while not gl.closed():
        with pcv.cuda_vertex_buffer as V:
            
            V[..., 0] = 0
            V[..., 1] = 0
            V[..., 2] = 0

            V[0,0] = .5
            V[0,1] = .5
            V[0,2] = .5


        pcv.draw(len(V), gl.width, gl.height)
        gl.swap()


if __name__ == "__main__":
    sys.exit(main())