namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: CYYY97S1_SERVICE_AUTHORIZATION   Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:40:32
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class CYYY97S1 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYYY97S1_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYYY97S1_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY9001_OA Cyyy9001Oa;
    GEN.ORT.YYY.IXX1B091_OA Ixx1b091Oa;
    GEN.ORT.YYY.ISC10011_OA Isc10011Oa;
    GEN.ORT.YYY.CYYY9831_IA Cyyy9831Ia;
    GEN.ORT.YYY.CYYY9831_OA Cyyy9831Oa;
    GEN.ORT.YYY.ISC11021_OA Isc11021Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020141_esc_flag;
    bool func_0020971826_esc_flag;
    bool func_0020971827_esc_flag;
    //       +->   CYYY97S1_SERVICE_AUTHORIZATION    01/09/2024  13:40
    //       !       EXPORTS:
    //       !         Work View exp_reference iyy1_server_data (Transient,
    //       !                     Export only)
    //       !           server_date
    //       !           server_time
    //       !           reference_id
    //       !           server_timestamp
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Work View loc_token_exists dont_change_text
    //       !           text_1
    //       !         Work View loc_exp_sec_token_inf ixx1_supplied
    //       !           flag
    //       !         Work View loc_error isc1_component
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !         Work View loc dont_change_return_codes
    //       !           1_ok
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 16-09-2005 New release
    //     2 !  
    //     3 !  SET loc dont_change_return_codes 1_ok TO 1 
    //     4 !  SET loc_token_exists dont_change_text text_1 TO "Y" 
    //     5 !   
    //     6 !  USE cyyy9001_exception_hndlng_dflt
    //     6 !     WHICH EXPORTS: Work View exp_error iyy1_component FROM Work
    //     6 !              View exp_error iyy1_component
    //     7 !   
    //     8 !  USE ixx1b091_is_token_exists
    //     8 !     WHICH EXPORTS: Work View loc_exp_sec_token_inf
    //     8 !              ixx1_supplied FROM Work View exp ixx1_supplied
    //     9 !   
    //    10 !  +->IF loc_exp_sec_token_inf ixx1_supplied flag ^=
    //    10 !  !        loc_token_exists dont_change_text text_1
    //    11 !  !   
    //    12 !  !  USE isc11021_service_login_process_s
    //    12 !  !     WHICH EXPORTS: Work View loc_error isc1_component FROM
    //    12 !  !              Work View exp_error isc1_component
    //    13 !  !   
    //    14 !  !  +->IF loc_error isc1_component return_code < loc
    //    14 !  !  !        dont_change_return_codes 1_ok
    //    15 !  !  !  USE cyyy9831_mv_sc1_to_yy1
    //    15 !  !  !     WHICH IMPORTS: Work View loc_error isc1_component TO
    //    15 !  !  !              Work View imp_error isc1_component
    //    15 !  !  !     WHICH EXPORTS: Work View exp_error iyy1_component
    //    15 !  !  !              FROM Work View exp_error iyy1_component
    //    16<---------ESCAPE
    //    14 !  !  +--
    //    17 !  !   
    //    18 !  !  USE isc10011_service_auth_check_s
    //    18 !  !     WHICH EXPORTS: Work View loc_error isc1_component FROM
    //    18 !  !              Work View exp_error isc1_component
    //    19 !  !   
    //    20 !  !  +->IF loc_error isc1_component return_code < loc
    //    20 !  !  !        dont_change_return_codes 1_ok
    //    21 !  !  !  USE cyyy9831_mv_sc1_to_yy1
    //    21 !  !  !     WHICH IMPORTS: Work View loc_error isc1_component TO
    //    21 !  !  !              Work View imp_error isc1_component
    //    21 !  !  !     WHICH EXPORTS: Work View exp_error iyy1_component
    //    21 !  !  !              FROM Work View exp_error iyy1_component
    //    20 !  !  +--
    //    10 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public CYYY97S1(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:40:32";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "CYYY97S1_SERVICE_AUTHORIZATION";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	CYYY97S1_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020141_localAlloc( "CYYY97S1_SERVICE_AUTHORIZATION" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020141_init(  );
        f_22020141(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020141(  )
    {
      func_0022020141_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020141" );
      Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
      {
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    See the description for the purpose                             
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    RELEASE HISTORY                                                 
        //    01_00 16-09-2005 New release                                    
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000003" );
        WLa.LocDontChangeReturnCodesQ1Ok = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
        Globdata.GetStateData().SetLastStatementNumber( "0000000004" );
        WLa.LocTokenExistsDontChangeTextText1 = FixedStringAttr.ValueOf("Y", 1);
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
        
        Cyyy9001Oa = (GEN.ORT.YYY.CYYY9001_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9001).Assembly,
        	"GEN.ORT.YYY.CYYY9001_OA" ));
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9001).Assembly,
        	"GEN.ORT.YYY.CYYY9001",
        	"Execute",
        	null,
        	Cyyy9001Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
        WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentSeverityCode, 1);
        WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
        WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentOriginServid);
        WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentContextString, 512);
        WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentReturnCode);
        WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentReasonCode);
        WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentChecksum, 15);
        Cyyy9001Oa.FreeInstance(  );
        Cyyy9001Oa = null;
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
        
        Ixx1b091Oa = (GEN.ORT.YYY.IXX1B091_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.IXX1B091).Assembly,
        	"GEN.ORT.YYY.IXX1B091_OA" ));
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.IXX1B091).Assembly,
        	"GEN.ORT.YYY.IXX1B091",
        	"Execute",
        	null,
        	Ixx1b091Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
        WLa.LocExpSecTokenInfIxx1SuppliedFlag = FixedStringAttr.ValueOf(Ixx1b091Oa.ExpIxx1SuppliedFlag, 1);
        Ixx1b091Oa.FreeInstance(  );
        Ixx1b091Oa = null;
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
        if ( CompareExit.CompareTo(WLa.LocExpSecTokenInfIxx1SuppliedFlag, WLa.LocTokenExistsDontChangeTextText1) != 0 )
        {
          f_20971826(  );
        }
        
      }
      f_0022020141_esctag: 
      ;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    
    public void f_20971826(  )
    {
      func_0020971826_esc_flag = false;
      {
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
        
        Isc11021Oa = (GEN.ORT.YYY.ISC11021_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.ISC11021).Assembly,
        	"GEN.ORT.YYY.ISC11021_OA" ));
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.ISC11021).Assembly,
        	"GEN.ORT.YYY.ISC11021",
        	"Execute",
        	null,
        	Isc11021Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
        WLa.LocErrorIsc1ComponentSeverityCode = FixedStringAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentSeverityCode, 1);
        WLa.LocErrorIsc1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentRollbackIndicator, 1);
        WLa.LocErrorIsc1ComponentOriginServid = DoubleAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentOriginServid);
        WLa.LocErrorIsc1ComponentContextString = StringAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentContextString, 512);
        WLa.LocErrorIsc1ComponentReturnCode = IntAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentReturnCode);
        WLa.LocErrorIsc1ComponentReasonCode = IntAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentReasonCode);
        WLa.LocErrorIsc1ComponentChecksum = FixedStringAttr.ValueOf(Isc11021Oa.ExpErrorIsc1ComponentChecksum, 15);
        Isc11021Oa.FreeInstance(  );
        Isc11021Oa = null;
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000014" );
        if ( ((double) WLa.LocErrorIsc1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
        {
          f_20971827(  );
        }
        
        if ( func_0020971826_esc_flag )
        {
          goto f_0020971826_esctag;
        }
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
        
        Isc10011Oa = (GEN.ORT.YYY.ISC10011_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.ISC10011).Assembly,
        	"GEN.ORT.YYY.ISC10011_OA" ));
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.ISC10011).Assembly,
        	"GEN.ORT.YYY.ISC10011",
        	"Execute",
        	null,
        	Isc10011Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
        WLa.LocErrorIsc1ComponentSeverityCode = FixedStringAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentSeverityCode, 1);
        WLa.LocErrorIsc1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentRollbackIndicator, 1);
        WLa.LocErrorIsc1ComponentOriginServid = DoubleAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentOriginServid);
        WLa.LocErrorIsc1ComponentContextString = StringAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentContextString, 512);
        WLa.LocErrorIsc1ComponentReturnCode = IntAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentReturnCode);
        WLa.LocErrorIsc1ComponentReasonCode = IntAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentReasonCode);
        WLa.LocErrorIsc1ComponentChecksum = FixedStringAttr.ValueOf(Isc10011Oa.ExpErrorIsc1ComponentChecksum, 15);
        Isc10011Oa.FreeInstance(  );
        Isc10011Oa = null;
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000020" );
        if ( ((double) WLa.LocErrorIsc1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
        {
          Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
          
          Cyyy9831Ia = (GEN.ORT.YYY.CYYY9831_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
          	"GEN.ORT.YYY.CYYY9831_IA" ));
          Cyyy9831Oa = (GEN.ORT.YYY.CYYY9831_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
          	"GEN.ORT.YYY.CYYY9831_OA" ));
          Cyyy9831Ia.ImpErrorIsc1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentSeverityCode, 1);
          Cyyy9831Ia.ImpErrorIsc1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentRollbackIndicator, 1)
            ;
          Cyyy9831Ia.ImpErrorIsc1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIsc1ComponentOriginServid);
          Cyyy9831Ia.ImpErrorIsc1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIsc1ComponentContextString, 512);
          Cyyy9831Ia.ImpErrorIsc1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIsc1ComponentReturnCode);
          Cyyy9831Ia.ImpErrorIsc1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIsc1ComponentReasonCode);
          Cyyy9831Ia.ImpErrorIsc1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentChecksum, 15);
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
          IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
          	"GEN.ORT.YYY.CYYY9831",
          	"Execute",
          	Cyyy9831Ia,
          	Cyyy9831Oa );
          if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
            ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
            ErrorData.ErrorEncounteredNoErrorFound) )
          {
            throw new ABException();
          }
          Globdata.GetStateData().SetCurrentABId( "0022020141" );
          Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
          Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
          WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentSeverityCode, 1);
          WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentRollbackIndicator, 1)
            ;
          WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentOriginServid);
          WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentContextString, 512);
          WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentReturnCode);
          WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentReasonCode);
          WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentChecksum, 15);
          Cyyy9831Ia.FreeInstance(  );
          Cyyy9831Ia = null;
          Cyyy9831Oa.FreeInstance(  );
          Cyyy9831Oa = null;
        }
        
      }
      f_0020971826_esctag: 
      ;
      return;
    }
    
    public void f_20971827(  )
    {
      func_0020971827_esc_flag = false;
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
        
        Cyyy9831Ia = (GEN.ORT.YYY.CYYY9831_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
        	"GEN.ORT.YYY.CYYY9831_IA" ));
        Cyyy9831Oa = (GEN.ORT.YYY.CYYY9831_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
        	"GEN.ORT.YYY.CYYY9831_OA" ));
        Cyyy9831Ia.ImpErrorIsc1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentSeverityCode, 1);
        Cyyy9831Ia.ImpErrorIsc1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentRollbackIndicator, 1);
        Cyyy9831Ia.ImpErrorIsc1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIsc1ComponentOriginServid);
        Cyyy9831Ia.ImpErrorIsc1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIsc1ComponentContextString, 512);
        Cyyy9831Ia.ImpErrorIsc1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIsc1ComponentReturnCode);
        Cyyy9831Ia.ImpErrorIsc1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIsc1ComponentReasonCode);
        Cyyy9831Ia.ImpErrorIsc1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIsc1ComponentChecksum, 15);
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9831).Assembly,
        	"GEN.ORT.YYY.CYYY9831",
        	"Execute",
        	Cyyy9831Ia,
        	Cyyy9831Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( "CYYY97S1_SERVICE_AUTHORIZATION" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
        WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentSeverityCode, 1);
        WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
        WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentOriginServid);
        WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentContextString, 512);
        WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentReturnCode);
        WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentReasonCode);
        WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9831Oa.ExpErrorIyy1ComponentChecksum, 15);
        Cyyy9831Ia.FreeInstance(  );
        Cyyy9831Ia = null;
        Cyyy9831Oa.FreeInstance(  );
        Cyyy9831Oa = null;
        Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
        func_0020971827_esc_flag = true;
        func_0020971826_esc_flag = true;
        func_0022020141_esc_flag = true;
      }
      f_0020971827_esctag: 
      ;
      return;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020141_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.CYYY97S1_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.CYYY97S1_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020141" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020141_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WOa.ExpReferenceIyy1ServerDataServerDate = 00000000;
      WOa.ExpReferenceIyy1ServerDataServerTime = 00000000;
      WOa.ExpReferenceIyy1ServerDataReferenceId = "00000000000000000000";
      WOa.ExpReferenceIyy1ServerDataServerTimestamp = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

