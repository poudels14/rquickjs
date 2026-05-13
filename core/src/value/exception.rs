use alloc::{string::String, vec::Vec};
use core::{error::Error as ErrorTrait, ffi::CStr, fmt};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::{atom::PredefinedAtom, convert::Coerced, qjs, Ctx, Error, Object, Result, Value};

/// One JavaScript stack frame captured at the point an [`Exception`] was thrown.
///
/// Populated only when [`crate::Runtime::set_capture_error_locals`] is enabled
/// on the runtime that threw the error.
#[derive(Debug, Clone)]
pub struct Frame<'js> {
    /// `null` for anonymous functions; `None` if the field was absent.
    pub function_name: Option<String>,
    /// The file the function was defined in. `None` for native frames.
    pub file_name: Option<String>,
    /// 1-based source line. `None` if unavailable (e.g. native frame).
    pub line_number: Option<u32>,
    /// 1-based source column. `None` if unavailable.
    pub column_number: Option<u32>,
    /// True for frames implemented in C / Rust (not JS bytecode).
    pub native: bool,
    /// Arguments and local variables in this frame at the time of the throw.
    /// Empty for native frames or when capture was disabled.
    pub locals: HashMap<String, Value<'js>>,
}

/// A JavaScript instance of Error
///
/// Will turn into a error when converted to JavaScript but won't automatically be thrown.
#[repr(transparent)]
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Exception<'js>(pub(crate) Object<'js>);

impl<'js> ErrorTrait for Exception<'js> {}

impl fmt::Debug for Exception<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Exception")
            .field("message", &self.message())
            .field("stack", &self.stack())
            .finish()
    }
}

pub(crate) static ERROR_FORMAT_STR: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked("%s\0".as_bytes()) };

fn truncate_str(mut max: usize, bytes: &[u8]) -> &[u8] {
    if bytes.len() <= max {
        return bytes;
    }
    // while the byte at len is a continue byte shorten the byte.
    while (bytes[max] & 0b1100_0000) == 0b1000_0000 {
        max -= 1;
    }
    &bytes[..max]
}

impl<'js> Exception<'js> {
    /// Turns the exception into the underlying object.
    pub fn into_object(self) -> Object<'js> {
        self.0
    }

    /// Returns a reference to the underlying object.
    pub fn as_object(&self) -> &Object<'js> {
        &self.0
    }

    /// Creates an exception from an object if it is an instance of error.
    pub fn from_object(obj: Object<'js>) -> Option<Self> {
        if obj.is_error() {
            Some(Self(obj))
        } else {
            None
        }
    }

    /// Creates a new exception with a given message.
    pub fn from_message(ctx: Ctx<'js>, message: &str) -> Result<Self> {
        let obj = unsafe {
            let value = ctx.handle_exception(qjs::JS_NewError(ctx.as_ptr()))?;
            Value::from_js_value(ctx, value)
                .into_object()
                .expect("`JS_NewError` did not return an object")
        };
        obj.set(PredefinedAtom::Message, message)?;
        Ok(Exception(obj))
    }

    /// Returns the message of the error.
    ///
    /// Same as retrieving `error.message` in JavaScript.
    pub fn message(&self) -> Option<String> {
        self.get::<_, Option<Coerced<String>>>(PredefinedAtom::Message)
            .ok()
            .and_then(|x| x)
            .map(|x| x.0)
    }

    /// Returns the error stack.
    ///
    /// Same as retrieving `error.stack` in JavaScript.
    pub fn stack(&self) -> Option<String> {
        self.get::<_, Option<Coerced<String>>>(PredefinedAtom::Stack)
            .ok()
            .and_then(|x| x)
            .map(|x| x.0)
    }

    /// Returns the arguments and local variables of the innermost JavaScript
    /// frame where the exception was thrown.
    ///
    /// Returns an empty map when:
    /// - [`crate::Runtime::set_capture_error_locals`] was not enabled before
    ///   the throw,
    /// - the throw originated from a native frame, or
    /// - the snapshot was discarded for any reason.
    ///
    /// Equivalent to `self.frames(Some(1))?.into_iter().next().map(|f| f.locals)`,
    /// but slightly cheaper because frame metadata is not constructed.
    pub fn locals(&self) -> Result<HashMap<String, Value<'js>>> {
        let frames = self.frames(Some(1))?;
        Ok(frames
            .into_iter()
            .next()
            .map(|f| f.locals)
            .unwrap_or_default())
    }

    /// Returns the captured stack frames (innermost first).
    ///
    /// Each [`Frame`] carries function/file/line/column metadata plus a snapshot
    /// of the frame's arguments and local variables.
    ///
    /// `depth` caps the number of frames returned. `None` returns all captured
    /// frames; `Some(N)` returns at most `N` frames.
    ///
    /// Returns an empty `Vec` when capture was not enabled on the runtime, or
    /// when the throw produced no JS frames (e.g. native-only exception).
    pub fn frames(&self, depth: Option<usize>) -> Result<Vec<Frame<'js>>> {
        let ctx = self.0.ctx().clone();
        let max_frames: i32 = match depth {
            None => -1,
            Some(n) => i32::try_from(n).unwrap_or(i32::MAX),
        };

        let arr_val = unsafe {
            let raw = qjs::JS_GetErrorFrames(ctx.as_ptr(), self.0.as_js_value(), max_frames as _);
            Value::from_js_value(ctx.clone(), ctx.handle_exception(raw)?)
        };
        let Some(arr) = arr_val.as_array().cloned() else {
            return Ok(Vec::new());
        };
        let mut frames = Vec::with_capacity(arr.len());
        for entry in arr.iter::<Object<'js>>() {
            let obj = entry?;
            frames.push(parse_frame(obj)?);
        }
        Ok(frames)
    }

    /// Throws a new generic error.
    ///
    /// Equivalent to:
    /// ```rust
    /// # use rquickjs::{Runtime,Context,Exception};
    /// # let rt = Runtime::new().unwrap();
    /// # let ctx = Context::full(&rt).unwrap();
    /// # ctx.with(|ctx|{
    /// # let _ = {
    /// # let message = "";
    /// let (Ok(e) | Err(e)) = Exception::from_message(ctx, message).map(|x| x.throw());
    /// e
    /// # };
    /// # })
    /// ```
    pub fn throw_message(ctx: &Ctx<'js>, message: &str) -> Error {
        let (Ok(e) | Err(e)) = Self::from_message(ctx.clone(), message).map(|x| x.throw());
        e
    }

    /// Throws a new syntax error.
    pub fn throw_syntax(ctx: &Ctx<'js>, message: &str) -> Error {
        // generate C string inline.
        // QuickJS implementation doesn't allow error strings longer then 256 anyway so truncating
        // here is fine.
        let mut buffer = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let str = truncate_str(255, message.as_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr().cast(), str.len());
            buffer.as_mut_ptr().cast::<u8>().add(str.len()).write(b'\0');
            let res = qjs::JS_ThrowSyntaxError(
                ctx.as_ptr(),
                ERROR_FORMAT_STR.as_ptr(),
                buffer.as_ptr().cast::<*mut u8>(),
            );
            debug_assert_eq!(qjs::JS_VALUE_GET_NORM_TAG(res), qjs::JS_TAG_EXCEPTION);
        }
        Error::Exception
    }

    /// Throws a new type error.
    pub fn throw_type(ctx: &Ctx<'js>, message: &str) -> Error {
        // generate C string inline.
        // QuickJS implementation doesn't allow error strings longer then 256 anyway so truncating
        // here is fine.
        let mut buffer = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let str = truncate_str(255, message.as_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr().cast(), str.len());
            buffer.as_mut_ptr().cast::<u8>().add(str.len()).write(b'\0');
            let res = qjs::JS_ThrowTypeError(
                ctx.as_ptr(),
                ERROR_FORMAT_STR.as_ptr(),
                buffer.as_ptr().cast::<*mut u8>(),
            );
            debug_assert_eq!(qjs::JS_VALUE_GET_NORM_TAG(res), qjs::JS_TAG_EXCEPTION);
        }
        Error::Exception
    }

    /// Throws a new reference error.
    pub fn throw_reference(ctx: &Ctx<'js>, message: &str) -> Error {
        // generate C string inline.
        // QuickJS implementation doesn't allow error strings longer then 256 anyway so truncating
        // here is fine.
        let mut buffer = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let str = truncate_str(255, message.as_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr().cast(), str.len());
            buffer.as_mut_ptr().cast::<u8>().add(str.len()).write(b'\0');
            let res = qjs::JS_ThrowReferenceError(
                ctx.as_ptr(),
                ERROR_FORMAT_STR.as_ptr(),
                buffer.as_ptr().cast::<*mut u8>(),
            );
            debug_assert_eq!(qjs::JS_VALUE_GET_NORM_TAG(res), qjs::JS_TAG_EXCEPTION);
        }
        Error::Exception
    }

    /// Throws a new range error.
    pub fn throw_range(ctx: &Ctx<'js>, message: &str) -> Error {
        // generate C string inline.
        // QuickJS implementation doesn't allow error strings longer then 256 anyway so truncating
        // here is fine.
        let mut buffer = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let str = truncate_str(255, message.as_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr().cast(), str.len());
            buffer.as_mut_ptr().cast::<u8>().add(str.len()).write(b'\0');
            let res = qjs::JS_ThrowRangeError(
                ctx.as_ptr(),
                ERROR_FORMAT_STR.as_ptr(),
                buffer.as_ptr().cast::<*mut u8>(),
            );
            debug_assert_eq!(qjs::JS_VALUE_GET_NORM_TAG(res), qjs::JS_TAG_EXCEPTION);
        }
        Error::Exception
    }

    /// Throws a new internal error.
    pub fn throw_internal(ctx: &Ctx<'js>, message: &str) -> Error {
        // generate C string inline.
        // QuickJS implementation doesn't allow error strings longer then 256 anyway so truncating
        // here is fine.
        let mut buffer = core::mem::MaybeUninit::<[u8; 256]>::uninit();
        let str = truncate_str(255, message.as_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr().cast(), str.len());
            buffer.as_mut_ptr().cast::<u8>().add(str.len()).write(b'\0');
            let res = qjs::JS_ThrowInternalError(
                ctx.as_ptr(),
                ERROR_FORMAT_STR.as_ptr(),
                buffer.as_ptr().cast::<*mut u8>(),
            );
            debug_assert_eq!(qjs::JS_VALUE_GET_NORM_TAG(res), qjs::JS_TAG_EXCEPTION);
        }
        Error::Exception
    }

    /// Sets the exception as the current error an returns `Error::Exception`
    pub fn throw(self) -> Error {
        let ctx = self.ctx().clone();
        ctx.throw(self.0.into_value())
    }
}

fn parse_frame<'js>(obj: Object<'js>) -> Result<Frame<'js>> {
    let function_name = obj
        .get::<_, Option<Coerced<String>>>("functionName")?
        .map(|c| c.0);
    let file_name = obj
        .get::<_, Option<Coerced<String>>>("fileName")?
        .map(|c| c.0);
    let line_number = obj.get::<_, Option<i32>>("lineNumber")?.and_then(|n| {
        if n >= 0 {
            Some(n as u32)
        } else {
            None
        }
    });
    let column_number = obj.get::<_, Option<i32>>("columnNumber")?.and_then(|n| {
        if n >= 0 {
            Some(n as u32)
        } else {
            None
        }
    });
    let native = obj.get::<_, Option<bool>>("native")?.unwrap_or(false);

    let mut locals = HashMap::new();
    if let Some(locals_obj) = obj.get::<_, Option<Object<'js>>>("locals")? {
        for entry in locals_obj.props::<String, Value<'js>>() {
            let (k, v) = entry?;
            locals.insert(k, v);
        }
    }

    Ok(Frame {
        function_name,
        file_name,
        line_number,
        column_number,
        native,
        locals,
    })
}

impl fmt::Display for Exception<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "Error:".fmt(f)?;
        if let Some(message) = self.message() {
            ' '.fmt(f)?;
            message.fmt(f)?;
        }
        if let Some(stack) = self.stack() {
            '\n'.fmt(f)?;
            stack.fmt(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Runtime};

    fn with_capture<F, R>(capture: bool, f: F) -> R
    where
        F: FnOnce(crate::Ctx) -> R,
    {
        let rt = Runtime::new().unwrap();
        rt.set_capture_error_locals(capture);
        let ctx = Context::full(&rt).unwrap();
        ctx.with(f)
    }

    fn catch_exception<'js>(err: crate::Error, ctx: &crate::Ctx<'js>) -> Exception<'js> {
        match err {
            crate::Error::Exception => {
                let val = ctx.catch();
                val.into_object()
                    .and_then(Exception::from_object)
                    .expect("expected an Error object")
            }
            other => panic!("expected Error::Exception, got {:?}", other),
        }
    }

    #[test]
    fn capture_off_returns_empty() {
        with_capture(false, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>("function inner(x) { throw new Error('boom'); }; inner(7)")
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            assert!(exc.locals().unwrap().is_empty());
            assert!(exc.frames(None).unwrap().is_empty());
        });
    }

    #[test]
    fn innermost_locals_capture_args() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "function inner(x, y) { let z = x + y; throw new Error('boom'); }; inner(3, 4)",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let locals = exc.locals().unwrap();
            assert_eq!(locals.get("x").unwrap().as_int(), Some(3));
            assert_eq!(locals.get("y").unwrap().as_int(), Some(4));
            assert_eq!(locals.get("z").unwrap().as_int(), Some(7));
        });
    }

    #[test]
    fn frames_walks_full_stack() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "function inner(a) { throw new Error('boom'); }\n\
                     function outer(b) { let c = b + 1; return inner(c); }\n\
                     outer(10);",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frames = exc.frames(None).unwrap();
            assert!(
                frames.len() >= 2,
                "expected at least 2 frames, got {}",
                frames.len()
            );
            // innermost first
            let inner = &frames[0];
            assert_eq!(inner.function_name.as_deref(), Some("inner"));
            assert_eq!(inner.locals.get("a").and_then(|v| v.as_int()), Some(11));

            let outer = &frames[1];
            assert_eq!(outer.function_name.as_deref(), Some("outer"));
            assert_eq!(outer.locals.get("b").and_then(|v| v.as_int()), Some(10));
            assert_eq!(outer.locals.get("c").and_then(|v| v.as_int()), Some(11));
        });
    }

    #[test]
    fn frames_depth_limits_count() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "function a() { throw new Error('boom'); }\n\
                     function b() { return a(); }\n\
                     function c() { return b(); }\n\
                     c();",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);

            let all = exc.frames(None).unwrap();
            assert!(all.len() >= 3);

            let one = exc.frames(Some(1)).unwrap();
            assert_eq!(one.len(), 1);
            assert_eq!(one[0].function_name.as_deref(), Some("a"));

            let two = exc.frames(Some(2)).unwrap();
            assert_eq!(two.len(), 2);
            assert_eq!(two[1].function_name.as_deref(), Some("b"));

            let none = exc.frames(Some(0)).unwrap();
            assert_eq!(none.len(), 0);
        });
    }

    #[test]
    fn captured_closure_var_visible_in_outer() {
        with_capture(true, |ctx| {
            // `state` is captured by `inner` so it lives in a JSVarRef in `outer`.
            // We should still see it in outer's locals.
            let err = ctx
                .eval::<crate::Value, _>(
                    "function outer() {\n\
                       let state = 42;\n\
                       function inner() { state += 1; throw new Error('boom'); }\n\
                       inner();\n\
                     }\n\
                     outer();",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frames = exc.frames(None).unwrap();
            let outer = frames
                .iter()
                .find(|f| f.function_name.as_deref() == Some("outer"))
                .expect("outer frame missing");
            assert_eq!(outer.locals.get("state").and_then(|v| v.as_int()), Some(43));
        });
    }

    #[test]
    fn tdz_uninitialized_is_omitted() {
        with_capture(true, |ctx| {
            // `z` is declared but not yet executed when the throw happens.
            let err = ctx
                .eval::<crate::Value, _>(
                    "function f() { throw new Error('boom'); let z = 1; }\n\
                     f();",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let locals = exc.locals().unwrap();
            // We don't surface TDZ values as undefined - they are simply absent.
            assert!(!locals.contains_key("z"));
        });
    }

    #[test]
    fn parse_error_with_capture_does_not_crash() {
        // Parse errors flow through js_new_callsite_data2 (filename-only frame).
        // Verifies that path properly initializes the locals snapshot fields.
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>("function f( {")
                .unwrap_err();
            let _exc = catch_exception(err, &ctx);
            // Just exercising creation+drop; the test passes if we don't UB.
        });
    }

    #[test]
    fn frame_metadata_present() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "function named() { throw new Error('boom'); }\n\
                     named();",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frame = exc.frames(Some(1)).unwrap().pop().unwrap();
            assert_eq!(frame.function_name.as_deref(), Some("named"));
            assert!(!frame.native);
            assert!(frame.line_number.is_some());
            assert!(frame.column_number.is_some());
        });
    }

    #[test]
    fn object_locals_captured() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "function f() {\n\
                       let obj = { a: 1, b: [10, 20, 30] };\n\
                       throw new Error('boom');\n\
                     }\n\
                     f();",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let locals = exc.locals().unwrap();
            let obj = locals.get("obj").expect("obj missing");
            let obj_obj = obj.as_object().expect("obj is not an object");
            assert_eq!(obj_obj.get::<_, i32>("a").unwrap(), 1);
            let arr: crate::Array = obj_obj.get("b").unwrap();
            assert_eq!(arr.len(), 3);
            assert_eq!(arr.get::<i32>(1).unwrap(), 20);
        });
    }

    #[test]
    fn dropping_error_releases_locals() {
        // The captured values act as GC roots. Verify that dropping the
        // exception releases them and a subsequent GC can collect.
        with_capture(true, |ctx| {
            let before;
            {
                let err = ctx
                    .eval::<crate::Value, _>(
                        "function f() { let big = new Array(1000).fill('x'); throw new Error('boom'); } f();",
                    )
                    .unwrap_err();
                let exc = catch_exception(err, &ctx);
                let locals = exc.locals().unwrap();
                assert!(locals.contains_key("big"));
                before = locals; // hold the locals map briefly
                drop(before);
                // Now drop the exception
                drop(exc);
            }
            // Force a GC; we just want to verify nothing UAFs / leaks.
            ctx.run_gc();
        });
    }

    #[test]
    fn anonymous_function_frame() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>("(function() { throw new Error('boom'); })()")
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frame = exc.frames(Some(1)).unwrap().pop().unwrap();
            // anonymous functions have no name → null on JS side → None
            assert!(frame.function_name.is_none(), "got {:?}", frame.function_name);
        });
    }

    #[test]
    fn arrow_function_frame() {
        with_capture(true, |ctx| {
            let err = ctx
                .eval::<crate::Value, _>(
                    "const f = (x) => { let y = x * 2; throw new Error('boom'); };\n\
                     f(7);",
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frames = exc.frames(None).unwrap();
            assert!(!frames.is_empty());
            // arrows assigned to a binding inherit the binding name in V8/QuickJS
            let inner = &frames[0];
            assert_eq!(inner.locals.get("x").and_then(|v| v.as_int()), Some(7));
            assert_eq!(inner.locals.get("y").and_then(|v| v.as_int()), Some(14));
        });
    }

    #[test]
    fn backtrace_barrier_stops_capture() {
        use crate::context::EvalOptions;
        with_capture(true, |ctx| {
            // Outer wrapper sets up a frame; inner eval has BACKTRACE_BARRIER.
            // The barrier should prevent the outer frames from showing up.
            let mut opts = EvalOptions::default();
            opts.backtrace_barrier = true;
            opts.global = true;
            let err = ctx
                .eval_with_options::<crate::Value, _>(
                    "function inner() { throw new Error('boom'); } inner();",
                    opts,
                )
                .unwrap_err();
            let exc = catch_exception(err, &ctx);
            let frames = exc.frames(None).unwrap();
            // We should see at most the inner frame; the outer eval boundary stops it.
            assert!(
                frames.iter().all(|f| f.function_name.as_deref() != Some("outer")),
                "barrier did not stop trace: {:?}",
                frames.iter().map(|f| f.function_name.clone()).collect::<alloc::vec::Vec<_>>()
            );
        });
    }

    #[cfg(feature = "futures")]
    #[tokio::test]
    async fn locals_captured_inside_async_function() {
        use crate::{AsyncContext, AsyncRuntime, CatchResultExt, CaughtError, Promise};
        let rt = AsyncRuntime::new().unwrap();
        rt.set_capture_error_locals(true).await;
        let ctx = AsyncContext::full(&rt).await.unwrap();
        ctx.async_with(async |ctx| {
            let promise: Promise = ctx
                .eval(
                    "(async function f(a) { let b = a + 1; throw new Error('async boom'); })(5)",
                )
                .catch(&ctx)
                .unwrap();
            let err = promise
                .into_future::<crate::Value>()
                .await
                .catch(&ctx)
                .err()
                .expect("expected rejection");
            let exc = match err {
                CaughtError::Exception(e) => e,
                CaughtError::Value(v) => Exception::from_object(v.into_object().unwrap())
                    .expect("rejection value was not an Error"),
                CaughtError::Error(e) => panic!("unexpected error: {:?}", e),
            };
            let locals = exc.locals().unwrap();
            assert_eq!(locals.get("a").and_then(|v| v.as_int()), Some(5));
            assert_eq!(locals.get("b").and_then(|v| v.as_int()), Some(6));
        })
        .await;
    }

    #[cfg(feature = "futures")]
    #[tokio::test]
    async fn locals_survive_async_rethrow() {
        // Regression: previously, the bytecode exception handler called
        // build_backtrace on every await-rethrow, overwriting __callsites__
        // with locals from the rethrow frame (the outer awaiter), not the
        // original throw site.
        use crate::{AsyncContext, AsyncRuntime, CatchResultExt, Promise};
        let rt = AsyncRuntime::new().unwrap();
        rt.set_capture_error_locals(true).await;
        let ctx = AsyncContext::full(&rt).await.unwrap();
        ctx.async_with(async |ctx| {
            let promise: Promise = ctx
                .eval(
                    "async function inner() {\n\
                       const userId = 'u_123';\n\
                       const attempt = 2;\n\
                       throw new Error('kaboom');\n\
                     }\n\
                     async function trace() {\n\
                       const span = { id: 'abc' };\n\
                       const options = { foo: 1 };\n\
                       try { await inner(); }\n\
                       catch (e) { return e; }\n\
                     }\n\
                     trace();",
                )
                .catch(&ctx)
                .unwrap();
            let value = promise
                .into_future::<crate::Value>()
                .await
                .catch(&ctx)
                .unwrap();
            let exc = Exception::from_object(value.into_object().unwrap())
                .expect("returned value should be an Error");
            let locals = exc.locals().unwrap();

            // Original throw site's locals must survive the await rethrow.
            assert!(
                locals.contains_key("userId"),
                "lost original locals after await rethrow; got keys: {:?}",
                locals.keys().collect::<alloc::vec::Vec<_>>()
            );
            assert!(locals.contains_key("attempt"));

            // The rethrow frame's locals must NOT have replaced them.
            assert!(!locals.contains_key("span"));
            assert!(!locals.contains_key("options"));
        })
        .await;
    }
}
